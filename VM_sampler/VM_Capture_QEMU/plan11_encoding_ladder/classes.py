#!/usr/bin/env python3
"""classes.py -- the class mapping file, its validator, the rewrite of cells.csv, the join table,
the class-only letter sequence and the inherited grid selection (SPEC_DETECTION.md section 2,
builder A).

Class membership comes from the author's mapping file ``inputs/classes.csv`` (path prefix ->
class, member index, sub-family letter); the code reads the file and never guesses a class from a
name. After ``apply`` every ``cell_id`` in ``cells.csv`` is a public identifier
(``sandbox_member_<m>__rep<rr>__<campaign>``), so no later file keyed by ``cell_id`` carries a
sandbox workload name.

Citation for the whole module: P3 0a (members by index, sub-families by letter, agents see
letters only); P3 D4 and ML 3.2 (the realized order as a class-only letter sequence); CR3 1.8
(deposit index i equals paper member i); CR3 2.31 (the re-launched control's workload key).

Corrections from the reviews of SPEC_DETECTION.md applied here (each marked "must change"):
  al-Kindi 2.13   refusal strings 9 and 17 quote row numbers only, never the author's prefix;
  al-Farabi M1    the campaign token of a sandbox or external cell id is ``--campaign-label`` or
                  the literal ``stage1``, never the raw launch label;
  al-Farabi M10   ``inherit-selection`` writes ``gates/detection/inherit_selection.json``;
  al-Farabi M11   ``apply`` refuses when ``extract/`` holds directories not keyed by a public id;
  al-Kindi 2.6    ``apply`` extends an existing ``inputs/head_drop.csv`` with a zero row per
                  workload key of every class (the head drop is declared, never derived).

No server path appears here. No sandbox workload is named here.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plan11_encoding_ladder import schema  # noqa: E402
from plan11_encoding_ladder import series as S  # noqa: E402
from plan11_encoding_ladder import verdicts as V  # noqa: E402

CITATION = ("P3 0a (members by index, sub-families by letter); P3 D4 and ML 3.2 (the realized order as a "
            "class-only letter sequence); CR3 1.8 (deposit index i = paper member i); CR3 2.31 (the re-launched "
            "control's workload key); SPEC_DETECTION.md section 2")

CLASSES = ("benign_kernel", "benign_relaunched", "benign_breadth", "idle", "harness_idle", "sandbox", "external")
BENIGN_CLASSES = ("benign_kernel", "benign_relaunched", "benign_breadth", "idle", "harness_idle")
POSITIVE_CLASSES = ("sandbox", "external")
TEST_ONLY_CLASSES = ("external",)
UNASSIGNED = "unassigned"
CLASS_LETTER = {"sandbox": "S", "benign_kernel": "B", "benign_breadth": "B", "idle": "I",
                "harness_idle": "H", "benign_relaunched": "R", "external": "X"}
ROLE_OF_CLASS = {"benign_kernel": "kernel", "idle": "idle", "sandbox": "sandbox", "benign_relaunched": "relaunched",
                 "benign_breadth": "breadth", "harness_idle": "harness_idle", "external": "external"}
KERNEL_FAMILY_RULE = "tier"          # section 7 item 3
RELAUNCHED_GROUPING = "parent"       # section 7 item 4
DEFAULT_CAMPAIGN_TOKEN = "stage1"    # al-Farabi M1: the campaign token of a sandbox or external id without --campaign-label
DEFAULT_GRID_ID = "W8_H4"

CLASSES_COLUMNS = ("path_prefix", "class", "member_index", "subfamily_letter", "rep", "order_index", "family", "workload_key")
JOIN_COLUMNS = ("cell_id", "class", "y", "workload_key", "family", "member_index", "subfamily_letter", "rep",
                "campaign", "order_index", "order_token", "split_role")
VALIDATION_SCHEMA = "plan11.detection.classes_validation.v1"
JOIN_SCHEMA = "plan11.detection.cell_classes.v1"
INHERIT_SCHEMA = "plan11.detection.inherit_selection.v1"
SELECTION_SCHEMA = "plan11.selection.v1"


# --------------------------------------------------------------------------- paths

def det_dir(out: Path) -> Path:
    return Path(out) / "gates" / "detection"


def classes_path(out: Path) -> Path:
    return Path(out) / "inputs" / "classes.csv"


def join_path(out: Path) -> Path:
    return det_dir(out) / "cell_classes.csv"


# --------------------------------------------------------------------------- the class file (2.1)

def read_classes_csv(path: Path) -> tuple[list[str], list[dict]]:
    """The mapping file as (header, rows); each row carries ``_row`` (1 for the first data row).
    Values are stripped; a trailing ``/`` of ``path_prefix`` is removed (SPEC_DETECTION 2.1)."""
    with Path(path).open(newline="") as f:
        rd = csv.reader(f)
        header = [h.strip() for h in next(rd, [])]
        rows = []
        for i, vals in enumerate(rd, start=1):
            if not any(v.strip() for v in vals):
                continue
            r = {h: (vals[j].strip() if j < len(vals) else "") for j, h in enumerate(header)}
            r["_row"] = i
            if "path_prefix" in r:
                r["path_prefix"] = r["path_prefix"].rstrip("/")
            rows.append(r)
    return header, rows


def _tail4(path: str) -> str:
    parts = [p for p in Path(str(path)).parts if p not in ("", "/")]
    return "/".join(parts[-4:])


def _matches(prefix: str, cell_path: str) -> bool:
    """``cell.path == prefix``, ``cell.path`` starts with ``prefix + "/"``, ``tail4 == prefix`` or ``tail4``
    starts with ``prefix + "/"`` (SPEC_DETECTION 2.1)."""
    if not prefix:
        return False
    cp = str(cell_path).rstrip("/")
    t4 = _tail4(cp)
    return cp == prefix or cp.startswith(prefix + "/") or t4 == prefix or t4.startswith(prefix + "/")


def match_cells(rows: list[dict], cells: list[dict]) -> tuple[dict, dict, list]:
    """Returns (winner_by_path, matches_by_row, ties): the winning row per cell path (the longest
    matching ``path_prefix``), the cell paths matched by each row index, and the (path, row a,
    row b) triples where two distinct prefixes of equal length both win."""
    winner: dict = {}
    by_row: dict = {i: [] for i in range(len(rows))}
    ties = []
    for c in cells:
        cp = str(c.get("path", ""))
        cands = [(len(r["path_prefix"]), i) for i, r in enumerate(rows) if _matches(r["path_prefix"], cp)]
        for _, i in cands:
            by_row[i].append(cp)
        if not cands:
            continue
        cands.sort(key=lambda t: (-t[0], t[1]))
        top_len = cands[0][0]
        tops = [i for ln, i in cands if ln == top_len]
        if len(tops) > 1:
            ties.append((cp, tops[0], tops[1]))
        winner[cp] = {"row": tops[0], "all": [i for _, i in cands]}
    return winner, by_row, ties


def _is_pos_int(s: str) -> bool:
    try:
        return int(s) >= 1 and str(int(s)) == s.strip()
    except (TypeError, ValueError):
        return False


def _campaign_defaulted(rows: list[dict], winner: dict, cells: list[dict], campaign_label) -> int:
    n = 0
    for c in cells:
        w = winner.get(str(c.get("path", "")))
        if w is None:
            continue
        if rows[w["row"]].get("class") in POSITIVE_CLASSES and not campaign_label:
            n += 1
    return n


def validate_classes(path: Path, cells: list[dict], *, relaunched_grouping: str = RELAUNCHED_GROUPING,
                     campaign_label: str | None = None, out: Path | None = None) -> dict:
    """The validator of ``inputs/classes.csv`` (SPEC_DETECTION 2.2, items 1 to 18, with al-Kindi
    2.13's row-number-only strings for items 9 and 17 and al-Farabi M1's campaign warning).
    Returns {status: ok | refused, refusals: [...], warnings: [...], unmatched_rows, unassigned_cells,
    counts: {class: n}, members: {index: letter}, subfamilies: {letter: [indices]}}. A row number
    counts data rows from 1. ``cells`` are the rows of ``cells.csv`` (every status). A refusal
    string quotes only row numbers, class values and column names. Citation: P3 0a; CR3 1.8;
    CR3 2.31 (item 14 and 15 under ``relaunched_grouping = "parent"``)."""
    refusals: list[str] = []
    warnings: list[str] = []
    header, rows = read_classes_csv(path)
    for h in header:
        if h not in CLASSES_COLUMNS:
            refusals.append(f"refused: unknown column {h}")
    for h in CLASSES_COLUMNS:
        if h not in header:
            refusals.append(f"refused: missing column {h}")
    if refusals:
        return {"status": "refused", "refusals": refusals, "warnings": warnings, "unmatched_rows": 0,
                "unassigned_cells": len(cells), "counts": {}, "members": {}, "subfamilies": {}}
    kernel_names = set(schema.KERNEL_NAMES)
    # per-row checks
    for r in rows:
        n = r["_row"]
        cls = r["class"]
        if cls not in CLASSES:
            refusals.append(f"refused: unknown class {cls} in row {n}")
            continue
        if ".." in Path(r["path_prefix"]).parts:
            refusals.append(f'refused: path_prefix contains ".." in row {n}')
        if cls == "sandbox":
            if not r["member_index"]:
                refusals.append(f"refused: sandbox row {n} without member_index")
            if not r["subfamily_letter"]:
                refusals.append(f"refused: sandbox row {n} without subfamily_letter")
        if cls == "external" and not r["member_index"]:
            refusals.append(f"refused: external row {n} without member_index")
        if r["member_index"] and not _is_pos_int(r["member_index"]):
            refusals.append(f"refused: member_index not a positive integer in row {n}")
        if r["subfamily_letter"] and r["subfamily_letter"] != "-" and not (len(r["subfamily_letter"]) == 1 and "A" <= r["subfamily_letter"] <= "Z"):
            refusals.append(f"refused: subfamily_letter not one capital letter in row {n}")
        if r["rep"]:
            try:
                rv = int(r["rep"])
                if not 0 <= rv <= 7:
                    raise ValueError
            except ValueError:
                refusals.append(f"refused: rep outside 0..7 in row {n}")
        if r["order_index"] and not _is_pos_int(r["order_index"]):
            refusals.append(f"refused: order_index not a positive integer in row {n}")
        if cls == "benign_breadth" and not r["family"]:
            refusals.append(f"refused: benign_breadth row {n} without family")
        if cls == "benign_relaunched" and relaunched_grouping == "parent":
            if not r["workload_key"]:
                refusals.append(f"refused: benign_relaunched row {n} without workload_key (parent kernel; CR3 2.31)")
            elif r["workload_key"] not in kernel_names:
                refusals.append(f"refused: workload_key {r['workload_key']} of benign_relaunched row {n} is not a kernel name")
    # duplicates of order_index and of path_prefix
    seen_prefix: dict = {}
    for r in rows:
        k = r["path_prefix"]
        if k in seen_prefix:
            refusals.append(f"refused: duplicate path_prefix in rows {seen_prefix[k]} and {r['_row']}")
        else:
            seen_prefix[k] = r["_row"]
    seen_order: dict = {}
    for r in rows:
        if r["order_index"] and _is_pos_int(r["order_index"]):
            o = int(r["order_index"])
            if o in seen_order:
                refusals.append(f"refused: duplicate order_index {o}")
            seen_order[o] = r["_row"]
    # member -> letter, sandbox against external
    members: dict = {}
    member_class: dict = {}
    for r in rows:
        if r["class"] in POSITIVE_CLASSES and _is_pos_int(r["member_index"]):
            m = int(r["member_index"])
            letter = r["subfamily_letter"] or "-"
            if r["class"] == "sandbox":
                if m in members and members[m] != letter:
                    refusals.append(f"refused: member {m} mapped to two sub-families ({members[m]}, {letter})")
                members.setdefault(m, letter)
            prev = member_class.get(m)
            if prev is not None and prev != r["class"]:
                refusals.append(f"refused: member_index {m} used by both sandbox and external rows")
            member_class.setdefault(m, r["class"])
    # matching
    winner, by_row, ties = match_cells(rows, cells)
    for cp, a, b in ties:
        refusals.append(f"refused: duplicate path_prefix in rows {rows[a]['_row']} and {rows[b]['_row']}")
    per_cell_cols = ("rep", "order_index")
    for i, r in enumerate(rows):
        for col in per_cell_cols:
            if r[col] and len(by_row[i]) > 1:
                refusals.append(f"refused: per-cell column {col} on a row that matches {len(by_row[i])} cells (row {r['_row']})")
    # item 17: two sandbox workload-level rows sharing a cell with different members
    wl_rows = [i for i, r in enumerate(rows) if r["class"] == "sandbox" and not r["rep"] and not r["order_index"]]
    reported = set()
    for cp, w in winner.items():
        idx = [i for i in w["all"] if i in wl_rows]
        for a in idx:
            for b in idx:
                if a < b and rows[a]["member_index"] != rows[b]["member_index"] and (a, b) not in reported:
                    reported.add((a, b))
                    refusals.append(f"refused: two members of one workload path in rows {rows[a]['_row']} and {rows[b]['_row']}")
    # item 8: a per-cell row (rep or order_index filled) must agree with the workload-level rows
    # that match the same cell; two workload-level rows of different length are the ordinary
    # hierarchy (the longer wins) and are not compared.
    reported8 = set()
    for cp, w in winner.items():
        win = rows[w["row"]]
        if not (win["rep"] or win["order_index"]):
            continue
        wl = [i for i in w["all"] if i != w["row"] and not (rows[i]["rep"] or rows[i]["order_index"])]
        if not wl:
            continue
        longest = max(len(rows[i]["path_prefix"]) for i in wl)
        for i in wl:
            if len(rows[i]["path_prefix"]) != longest or (w["row"], i) in reported8:
                continue
            other = rows[i]
            for col in ("class", "member_index", "subfamily_letter", "family", "workload_key"):
                a, b = win.get(col, ""), other.get(col, "")
                if a and b and a != b:
                    reported8.add((w["row"], i))
                    refusals.append(f"refused: rows {win['_row']} and {other['_row']} disagree on {col} for one cell")
                    break
    # item 16: roles
    cells_by_path = {str(c.get("path", "")): c for c in cells}
    for cp, w in winner.items():
        r = rows[w["row"]]
        c = cells_by_path[cp]
        role = str(c.get("role", ""))
        if r["class"] == "benign_kernel" and not (role == "kernel" and str(c.get("kernel")) in kernel_names):
            refusals.append(f"refused: benign_kernel row {r['_row']} matches a cell whose role is {role}")
        if r["class"] == "idle" and role not in ("idle", "unknown"):
            refusals.append(f"refused: idle row {r['_row']} matches a cell whose role is {role}")
    # M1 warning
    nd = _campaign_defaulted(rows, winner, cells, campaign_label)
    if nd:
        warnings.append(f"campaign token defaulted to {DEFAULT_CAMPAIGN_TOKEN} for {nd} rows")
    unmatched = sum(1 for i in range(len(rows)) if not by_row[i])
    counts: dict = {}
    for cp, w in winner.items():
        c = cells_by_path[cp]
        if c.get("status", "ok") in (schema.STATUS_OK, schema.STATUS_UNKNOWN_KERNEL):
            cls = rows[w["row"]]["class"]
            counts[cls] = counts.get(cls, 0) + 1
    unassigned = sum(1 for c in cells if str(c.get("path", "")) not in winner)
    subfam: dict = {}
    for m, letter in sorted(members.items()):
        subfam.setdefault(letter, []).append(m)
    # unique refusals, order kept
    seen = set()
    uniq = [x for x in refusals if not (x in seen or seen.add(x))]
    return {"status": "refused" if uniq else "ok", "refusals": uniq, "warnings": warnings,
            "unmatched_rows": unmatched, "unassigned_cells": unassigned, "counts": counts,
            "members": {str(m): l for m, l in sorted(members.items())}, "subfamilies": subfam}


def write_validation(out: Path, result: dict, classes_csv: Path, params: dict | None = None) -> Path:
    p = det_dir(out) / "classes_validation.json"
    prm = {"classes_csv": str(classes_csv), "inputs_sha256": S.inputs_sha256([classes_csv, Path(out) / "cells.csv"], Path(out))}
    prm.update(params or {})
    return S.write_json(p, VALIDATION_SCHEMA, prm, CITATION, result)


# --------------------------------------------------------------------------- apply (2.3)

def _plan11_rep(c: dict) -> int:
    try:
        return int(float(c.get("rep") or 0))
    except (TypeError, ValueError):
        return 0


def _rewrite_row(c: dict, r: dict, campaign_label: str | None) -> dict:
    """The rewrite of one matched cells.csv row by class (SPEC_DETECTION 2.3, table). The campaign
    token of a sandbox or external row is ``campaign_label`` or ``stage1`` (al-Farabi M1)."""
    cls = r["class"]
    new = dict(c)
    rr = int(r["rep"]) if r.get("rep") else _plan11_rep(c)
    if cls == "benign_kernel":
        return new
    campaign = str(c.get("campaign") or schema.campaign_of(str(c.get("label") or "")))
    if cls in POSITIVE_CLASSES:
        campaign = campaign_label or DEFAULT_CAMPAIGN_TOKEN
        new["campaign"] = campaign
    new["role"] = ROLE_OF_CLASS[cls]
    new["rep"] = rr
    if cls == "idle":
        new["kernel"] = "idle"; new["archetype_predicted"] = "control"; new["cell_id"] = f"idle__rep{rr:02d}__{campaign}"
    elif cls == "sandbox":
        m = int(r["member_index"])
        new["kernel"] = f"sandbox_member_{m}"; new["archetype_predicted"] = "sandbox"
        new["cell_id"] = f"sandbox_member_{m}__rep{rr:02d}__{campaign}"
    elif cls == "benign_relaunched":
        wk = r.get("workload_key") or str(c.get("kernel"))
        new["kernel"] = f"relaunched_{wk}"; new["archetype_predicted"] = "relaunched"
        new["cell_id"] = f"relaunched_{wk}__rep{rr:02d}__{campaign}"
    elif cls == "benign_breadth":
        new["archetype_predicted"] = "breadth"
    elif cls == "harness_idle":
        new["kernel"] = "harness_idle"; new["archetype_predicted"] = "control"; new["cell_id"] = f"harness_idle__rep{rr:02d}__{campaign}"
    elif cls == "external":
        m = int(r["member_index"])
        new["kernel"] = f"external_member_{m}"; new["archetype_predicted"] = "external"
        new["cell_id"] = f"external_member_{m}__rep{rr:02d}__{campaign}"
    if new.get("status") == schema.STATUS_UNKNOWN_KERNEL:
        new["status"] = schema.STATUS_OK
    return new


def _extract_dirs_not_public(out: Path, public_ids: set) -> list[str]:
    ed = Path(out) / "extract"
    if not ed.is_dir():
        return []
    return sorted(d.name for d in ed.iterdir() if d.is_dir() and d.name not in public_ids)


def apply_classes(out: Path, classes_csv: Path | None = None, *, campaign_label: str | None = None,
                  relaunched_grouping: str = RELAUNCHED_GROUPING, kernel_family_rule: str = KERNEL_FAMILY_RULE) -> dict:
    """``classes apply`` (SPEC_DETECTION 2.3): validates, copies ``cells.csv`` to
    ``cells.pre_classes.csv`` once, rewrites every matched row (role, kernel, archetype_predicted,
    cell_id, status by the table of 2.3; the campaign token of sandbox and external rows is
    ``campaign_label`` or ``stage1``, al-Farabi M1), records ``classes_applied`` in
    ``cells.index.json``, then builds the join (2.4) and the letter sequence (2.5). Refuses, with
    ``cells.csv`` untouched, unless validation is ``ok``, and (al-Farabi M11) when ``extract/`` holds
    directories not keyed by a public cell id after the rewrite. Idempotent: matching uses ``path``.
    Returns the validation dict with ``applied`` and ``cells_csv_sha256``. Citation: P3 0a; CR3 1.8."""
    out = Path(out)
    target = classes_path(out)
    if classes_csv is not None and Path(classes_csv).resolve() != target.resolve():
        src = Path(classes_csv)
        if not src.is_file():
            raise FileNotFoundError(str(src))
        if target.is_file() and S.sha256_file(target) != S.sha256_file(src):
            res = {"status": "refused", "refusals": [V.refused("inputs/classes.csv exists; edit it or pass --force")],
                   "warnings": [], "applied": False}
            write_validation(out, res, target)
            return res
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file():
            shutil.copyfile(src, target)
    if not target.is_file():
        raise FileNotFoundError(str(target))
    cells_csv_p = out / "cells.csv"
    if not cells_csv_p.is_file():
        raise FileNotFoundError(str(cells_csv_p))
    cells = S.read_csv(cells_csv_p)
    res = validate_classes(target, cells, relaunched_grouping=relaunched_grouping, campaign_label=campaign_label, out=out)
    params = {"campaign_label": campaign_label, "relaunched_grouping": relaunched_grouping,
              "kernel_family_rule": kernel_family_rule, "default_campaign_token": DEFAULT_CAMPAIGN_TOKEN}
    res["applied"] = False
    if res["status"] != "ok":
        write_validation(out, res, target, params)
        return res
    _, rows = read_classes_csv(target)
    winner, _, _ = match_cells(rows, cells)
    new_rows = []
    for c in cells:
        w = winner.get(str(c.get("path", "")))
        new_rows.append(_rewrite_row(c, rows[w["row"]], campaign_label) if w else dict(c))
    stray = _extract_dirs_not_public(out, {r["cell_id"] for r in new_rows})
    if stray:
        res["status"] = "refused"
        res["refusals"] = [V.refused(f"extract/ holds {len(stray)} directories not keyed by a public cell_id; remove them and re-run")]
        write_validation(out, res, target, params)
        return res
    pre = out / "cells.pre_classes.csv"
    if not pre.is_file():
        shutil.copyfile(cells_csv_p, pre)
    cols = list(schema.CELLS_COLUMNS)
    tmp = cells_csv_p.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in new_rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    tmp.replace(cells_csv_p)
    ip = out / "cells.index.json"
    try:
        idx = json.loads(ip.read_text()) if ip.is_file() else {"schema": schema.CELLS_SCHEMA, "params": {}, "citation": CITATION}
    except ValueError:
        idx = {"schema": schema.CELLS_SCHEMA, "params": {}, "citation": CITATION}
    idx["classes_applied"] = {"classes_csv_sha256": S.sha256_file(target), "counts": res["counts"],
                              "campaign_label": campaign_label, "default_campaign_token": DEFAULT_CAMPAIGN_TOKEN}
    ip.write_text(json.dumps(idx, indent=1, default=S._json_default))
    res["applied"] = True
    res["cells_csv_sha256"] = S.sha256_file(cells_csv_p)
    write_validation(out, res, target, params)
    join = build_join(out, kernel_family_rule=kernel_family_rule, relaunched_grouping=relaunched_grouping, campaign_label=campaign_label)
    write_letter_sequence(out, join)
    extend_head_drop(out, join)
    return res


def extend_head_drop(out: Path, join: list[dict]) -> Path | None:
    """When ``inputs/head_drop.csv`` exists, append one row ``(workload_key, 0, "default 0; declared at
    D2")`` for every workload key of every class that the file does not list (al-Kindi 2.6; al-Farabi
    M5: the head drop is a declared constant per key, 0 unless the author declares a value before
    the data). Existing rows are never changed. Returns the path, or None when the file is absent."""
    p = Path(out) / "inputs" / "head_drop.csv"
    if not p.is_file():
        return None
    rows = S.read_csv(p)
    have = {r.get("kernel") for r in rows}
    keys = []
    for j in join:
        k = j.get("workload_key")
        if k and k not in have and k not in keys:
            keys.append(k)
    if not keys:
        return p
    for k in keys:
        rows.append({"kernel": k, "head_drop_pairs": 0, "reason": "default 0; declared at D2"})
    return S.write_csv(p, S.HEAD_DROP_COLUMNS, rows)


# --------------------------------------------------------------------------- the join (2.4)

def _family_of(cls: str, r: dict, c: dict, kernel_family_rule: str) -> str:
    if r.get("family"):
        return r["family"]
    if cls == "benign_kernel":
        return "kernels" if kernel_family_rule == "tier" else schema.ARCHETYPE_OF.get(str(c.get("kernel")), "unknown")
    return {"benign_relaunched": "relaunched", "idle": "idle", "harness_idle": "harness_idle",
            "sandbox": "sandbox", "external": "external"}.get(cls, str(c.get("family") or cls))


def _workload_key_of(cls: str, r: dict, c: dict, relaunched_grouping: str) -> str:
    if cls == "benign_kernel":
        return str(c.get("kernel"))
    if cls == "benign_relaunched":
        return r.get("workload_key") if (relaunched_grouping == "parent" and r.get("workload_key")) else str(c.get("kernel"))
    if cls == "benign_breadth":
        return r.get("workload_key") or str(c.get("kernel"))
    if cls == "idle":
        return "idle"
    if cls == "harness_idle":
        return "harness_idle"
    if cls == "sandbox":
        return f"sandbox_member_{int(r['member_index'])}"
    if cls == "external":
        return f"external_member_{int(r['member_index'])}"
    return str(c.get("kernel"))


def order_token(cls: str, member_index, rep) -> str:
    """``<L>[<m>]r<rep>`` with ``L = CLASS_LETTER[class]``, ``<m>`` for S and X only (SPEC_DETECTION 2.5)."""
    L = CLASS_LETTER[cls]
    m = f"{int(member_index)}" if L in ("S", "X") and member_index not in (None, "", 0) else ""
    return f"{L}{m}r{int(rep)}"


def build_join(out: Path, *, kernel_family_rule: str = KERNEL_FAMILY_RULE, relaunched_grouping: str = RELAUNCHED_GROUPING,
               campaign_label: str | None = None) -> list[dict]:
    """``gates/detection/cell_classes.csv`` and ``.json`` (SPEC_DETECTION 2.4): one row per ok cell of
    ``cells.csv`` (plus one per unassigned cell with ``class = unassigned`` and every derived column
    empty): cell_id, class, y (sandbox for the positive classes, benign for the five benign classes,
    ML 1.2), workload_key, family (the benign family for LOFO, G-FP and the per-family recall, ML 1.6
    item 4; ``kernels`` under ``kernel_family_rule = "tier"``, the predicted archetype under
    ``"archetype"``), member_index, subfamily_letter, rep, campaign, order_index, order_token,
    split_role (``test_only`` for ``external``, P3 0a stage 3). The JSON carries the counts per class,
    family, member and sub-family, S, B and ``n_assignments = comb(S + B, S)``. Citation: ML 1.2; P3 0a."""
    out = Path(out)
    cp = classes_path(out)
    if not cp.is_file():
        raise FileNotFoundError(str(cp))
    cells = S.read_csv(out / "cells.csv")
    _, rows = read_classes_csv(cp)
    winner, _, _ = match_cells(rows, cells)
    join: list[dict] = []
    for c in cells:
        w = winner.get(str(c.get("path", "")))
        if c.get("status", "ok") != schema.STATUS_OK:
            continue
        if w is None:
            join.append({"cell_id": c["cell_id"], "class": UNASSIGNED, "y": "", "workload_key": "", "family": "",
                         "member_index": "", "subfamily_letter": "", "rep": "", "campaign": "", "order_index": "",
                         "order_token": "", "split_role": ""})
            continue
        r = rows[w["row"]]
        cls = r["class"]
        rep = int(r["rep"]) if r.get("rep") else _plan11_rep(c)
        m = int(r["member_index"]) if r.get("member_index") else 0
        letter = (r.get("subfamily_letter") or "-") if cls in POSITIVE_CLASSES else "-"
        campaign = (campaign_label or DEFAULT_CAMPAIGN_TOKEN) if cls in POSITIVE_CLASSES else str(c.get("campaign") or "")
        oi = int(r["order_index"]) if r.get("order_index") else ""
        join.append({"cell_id": c["cell_id"], "class": cls, "y": "sandbox" if cls in POSITIVE_CLASSES else "benign",
                     "workload_key": _workload_key_of(cls, r, c, relaunched_grouping),
                     "family": _family_of(cls, r, c, kernel_family_rule), "member_index": m, "subfamily_letter": letter,
                     "rep": rep, "campaign": campaign, "order_index": oi,
                     "order_token": order_token(cls, m, rep), "split_role": "test_only" if cls in TEST_ONLY_CLASSES else "train_test"})
    S.write_csv(join_path(out), JOIN_COLUMNS, join)
    counts = {"class": {}, "family": {}, "member": {}, "subfamily": {}}
    for j in join:
        counts["class"][j["class"]] = counts["class"].get(j["class"], 0) + 1
        if j["class"] != UNASSIGNED:
            counts["family"][j["family"]] = counts["family"].get(j["family"], 0) + 1
        if j["class"] in POSITIVE_CLASSES:
            counts["member"][str(j["member_index"])] = counts["member"].get(str(j["member_index"]), 0) + 1
            counts["subfamily"][j["subfamily_letter"]] = counts["subfamily"].get(j["subfamily_letter"], 0) + 1
    adm = admissible_ids(out)
    S_n = len({j["member_index"] for j in join if j["class"] == "sandbox" and (adm is None or j["cell_id"] in adm)})
    B_n = len({j["workload_key"] for j in join if j["y"] == "benign" and (adm is None or j["cell_id"] in adm)})
    params = {"classes_csv_sha256": S.sha256_file(cp), "kernel_family_rule": kernel_family_rule,
              "relaunched_grouping": relaunched_grouping, "campaign_label": campaign_label,
              "default_campaign_token": DEFAULT_CAMPAIGN_TOKEN, "S_source": "admissibility.csv" if adm is not None else "ok cells (admissibility not run)",
              "inputs_sha256": S.inputs_sha256([cp, out / "cells.csv", det_dir(out) / "admissibility.csv"], out)}
    S.write_json(det_dir(out) / "cell_classes.json", JOIN_SCHEMA, params, CITATION,
                 {"n_rows": len(join), "counts": counts, "S": S_n, "B": B_n, "n_assignments": math.comb(S_n + B_n, S_n) if S_n + B_n else 0})
    return join


def admissible_ids(out: Path) -> set | None:
    """The admissible cell ids from ``gates/detection/admissibility.csv`` when it exists, else None."""
    p = det_dir(out) / "admissibility.csv"
    if not p.is_file():
        return None
    return {r["cell_id"] for r in S.read_csv(p) if str(r.get("admissible", "")).lower() == "true"}


def load_join(out: Path, *, include_unassigned: bool = False) -> list[dict]:
    """The join rows with ``member_index``, ``rep`` and ``order_index`` as integers (-1 for a missing
    order index, 0 for a non-member)."""
    p = join_path(out)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    rows = []
    for r in S.read_csv(p):
        if r["class"] == UNASSIGNED and not include_unassigned:
            continue
        r["member_index"] = int(float(r["member_index"])) if r.get("member_index") not in ("", None) else 0
        r["rep"] = int(float(r["rep"])) if r.get("rep") not in ("", None) else 0
        r["order_index"] = int(float(r["order_index"])) if r.get("order_index") not in ("", None) else -1
        rows.append(r)
    return rows


# --------------------------------------------------------------------------- the letter sequence (2.5)

def letter_sequence(join_rows: list[dict]) -> list[str]:
    """The class-only letter sequence (SPEC_DETECTION 2.5; P3 D4; ML 3.2): the tokens sorted by
    ``order_index``, one per classed cell; when any classed cell lacks ``order_index`` the single
    line ``not run: order_index missing for <n> cells``."""
    classed = [j for j in join_rows if j.get("class") and j["class"] != UNASSIGNED]
    missing = [j for j in classed if j.get("order_index") in ("", None, -1, "-1")]
    if missing:
        return [V.not_run(f"order_index missing for {len(missing)} cells")]
    ordered = sorted(classed, key=lambda j: int(j["order_index"]))
    return [j["order_token"] or order_token(j["class"], j.get("member_index") or 0, j.get("rep") or 0) for j in ordered]


def write_letter_sequence(out: Path, join_rows: list[dict] | None = None) -> Path:
    out = Path(out)
    if join_rows is None:
        join_rows = load_join(out)
    toks = letter_sequence(join_rows)
    d = det_dir(out)
    d.mkdir(parents=True, exist_ok=True)
    (d / "letter_sequence.txt").write_text(" ".join(toks) + "\n")
    if len(toks) == 1 and toks[0].startswith("not run:"):
        S.write_csv(d / "letter_sequence.csv", ("order_index", "token", "class"), [{"order_index": "", "token": toks[0], "class": ""}])
    else:
        ordered = sorted((j for j in join_rows if j.get("class") and j["class"] != UNASSIGNED), key=lambda j: int(j["order_index"]))
        S.write_csv(d / "letter_sequence.csv", ("order_index", "token", "class"),
                    [{"order_index": int(j["order_index"]), "token": t, "class": j["class"]} for j, t in zip(ordered, toks)])
    return d / "letter_sequence.txt"


# --------------------------------------------------------------------------- inherit-selection (2.7)

def inherit_selection(out: Path, *, from_path: Path | None = None, default_grid: str | None = None, force: bool = False) -> Path:
    """``gates/selection.json`` as a verbatim copy of the encoding run's selection (``params.inherited_from``,
    ``inherited_sha256``, ``grid_source = "inherited: <path>"``) or, with ``default_grid``, the default
    point for the five rungs (``grid_source = "default: <id> (no inherited selection)"``); the (W, H)
    per rung is inherited from the encoding paper's Table 5, fixed per encoding and never per class
    (ML 1.1; K3 2.2). Exactly one of ``from_path`` and ``default_grid`` is given. An existing file not
    written by this command (no ``params.grid_source``) is refused without ``force``; the outcome is
    written to ``gates/detection/inherit_selection.json`` (al-Farabi M10). Citation: ML 1.1; SPEC 3.5.7."""
    out = Path(out)
    sp = out / "gates" / "selection.json"
    rec = det_dir(out) / "inherit_selection.json"
    if (from_path is None) == (default_grid is None):
        raise ValueError("exactly one of from_path and default_grid")
    if sp.is_file() and not force:
        try:
            old = S.read_json(sp)
        except ValueError:
            old = {}
        if not (old.get("params") or {}).get("grid_source"):
            msg = V.refused("gates/selection.json exists and was written by gates_temporal select; pass --force to replace")
            S.write_json(rec, INHERIT_SCHEMA, {"from": str(from_path) if from_path else None, "default_grid": default_grid, "force": force},
                         CITATION, {"status": msg, "selection_json": str(sp)})
            return rec
    if from_path is not None:
        src = Path(from_path)
        if not src.is_file():
            raise FileNotFoundError(str(src))
        doc = S.read_json(src)
        payload = {k: v for k, v in doc.items() if k not in ("schema", "params", "citation")}
        params = dict(doc.get("params") or {})
        params.update({"inherited_from": str(src), "inherited_sha256": S.sha256_file(src), "grid_source": f"inherited: {src}"})
        grid_source = params["grid_source"]
    else:
        gid = str(default_grid)
        W, H = S.parse_grid_id(gid)
        payload = {r: {"grid_id": gid, "W": W, "H": H, "passes_acceptance": False,
                       "selected_by": "default: no inherited selection", "gates_passed": [], "refusal": ""} for r in S.RUNGS}
        grid_source = f"default: {gid} (no inherited selection)"
        params = {"grid_source": grid_source, "default_grid": gid}
    sp.parent.mkdir(parents=True, exist_ok=True)
    S.write_json(sp, SELECTION_SCHEMA, params, CITATION, payload)
    S.write_json(rec, INHERIT_SCHEMA, {"from": str(from_path) if from_path else None, "default_grid": default_grid, "force": force,
                                        "inputs_sha256": S.inputs_sha256([from_path] if from_path else [], out)},
                 CITATION, {"status": "ok", "grid_source": grid_source, "selection_json": str(sp), "rungs": sorted(payload)})
    return sp


def grid_source(out: Path) -> str:
    """``params.grid_source`` of ``gates/selection.json`` (``"selection.json"`` when the file was written
    by ``gates_temporal select``; ``"no selection"`` when absent)."""
    sp = Path(out) / "gates" / "selection.json"
    if not sp.is_file():
        return "no selection"
    try:
        doc = S.read_json(sp)
    except ValueError:
        return "no selection"
    return str((doc.get("params") or {}).get("grid_source") or "selection.json")


# --------------------------------------------------------------------------- CLI (2.8)

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="classes.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("validate")
    a.add_argument("--out", required=True); a.add_argument("--classes", required=True)
    a.add_argument("--relaunched-grouping", default=RELAUNCHED_GROUPING, choices=("parent", "own"))
    a.add_argument("--campaign-label", default=None)
    b = sub.add_parser("apply")
    b.add_argument("--out", required=True); b.add_argument("--classes", default=None)
    b.add_argument("--campaign-label", default=None)
    b.add_argument("--kernel-family-rule", default=KERNEL_FAMILY_RULE, choices=("tier", "archetype"))
    b.add_argument("--relaunched-grouping", default=RELAUNCHED_GROUPING, choices=("parent", "own"))
    c = sub.add_parser("join")
    c.add_argument("--out", required=True)
    c.add_argument("--kernel-family-rule", default=KERNEL_FAMILY_RULE, choices=("tier", "archetype"))
    c.add_argument("--relaunched-grouping", default=RELAUNCHED_GROUPING, choices=("parent", "own"))
    c.add_argument("--campaign-label", default=None)
    d = sub.add_parser("letter-sequence"); d.add_argument("--out", required=True)
    e = sub.add_parser("inherit-selection"); e.add_argument("--out", required=True)
    g = e.add_mutually_exclusive_group(required=True)
    g.add_argument("--from", dest="from_path", default=None); g.add_argument("--default", dest="default_grid", default=None)
    e.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    try:
        if args.cmd == "validate":
            cp = Path(args.classes)
            if not cp.is_file():
                print(f"missing input: {cp}", file=sys.stderr); return 2
            cells_p = out / "cells.csv"
            if not cells_p.is_file():
                print(f"missing input: {cells_p}", file=sys.stderr); return 2
            res = validate_classes(cp, S.read_csv(cells_p), relaunched_grouping=args.relaunched_grouping, campaign_label=args.campaign_label)
            p = write_validation(out, res, cp, {"relaunched_grouping": args.relaunched_grouping})
            print(f"[classes validate] {res['status']} ({len(res['refusals'])} refusals) -> {p}")
            return 0
        if args.cmd == "apply":
            res = apply_classes(out, Path(args.classes) if args.classes else None, campaign_label=args.campaign_label,
                                relaunched_grouping=args.relaunched_grouping, kernel_family_rule=args.kernel_family_rule)
            print(f"[classes apply] {res['status']}; applied = {res.get('applied')}; counts = {res.get('counts')}")
            return 0
        if args.cmd == "join":
            join = build_join(out, kernel_family_rule=args.kernel_family_rule, relaunched_grouping=args.relaunched_grouping,
                              campaign_label=args.campaign_label)
            print(f"[classes join] {len(join)} rows -> {join_path(out)}")
            return 0
        if args.cmd == "letter-sequence":
            p = write_letter_sequence(out)
            print(f"[classes letter-sequence] -> {p}")
            return 0
        if args.cmd == "inherit-selection":
            p = inherit_selection(out, from_path=Path(args.from_path) if args.from_path else None,
                                  default_grid=args.default_grid, force=args.force)
            print(f"[classes inherit-selection] -> {p}")
            return 0
    except FileNotFoundError as e:
        print(f"missing input: {e}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
