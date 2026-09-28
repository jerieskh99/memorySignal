"""classes.py: the validator (every refusal of SPEC_DETECTION 2.2), apply, the join, the letter
sequence, inherit-selection, the head-drop extension. Hand-built cells; no corpus."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

from _det_common import C, S, V, tmp_out, rows, jload, run_cli, corpus, corpus_root, PKG
from plan11_encoding_ladder import schema

HDR = "path_prefix,class,member_index,subfamily_letter,rep,order_index,family,workload_key"


def _cell(path, kernel, role, rep, seed, status="ok", label="synth"):
    camp = schema.campaign_of(label)
    cid = schema.cell_id_of(kernel, role, rep, camp)
    return {"cell_id": cid, "kernel": kernel, "role": role, "archetype_predicted": schema.ARCHETYPE_OF.get(kernel, "control" if role == "idle" else "unknown"),
            "seed": seed, "rep": rep, "rep_dir": 1, "label": label, "campaign": camp, "path": path, "traj_file": "t.csv.zst", "status": status}


def make_cells(n_members: int = 3, reps: int = 2, kernels=("gemm", "gibbs"), idle: int = 2) -> list[dict]:
    cells = []
    for k in kernels:
        for r in range(reps):
            cells.append(_cell(f"/r/kernel/kernel_{k}_v2/--seed_{42 if r == 0 else 1000 * r}_--duration_40/rep001__synth", k, "kernel", r, 42 if r == 0 else 1000 * r))
    for r in range(idle):
        cells.append(_cell(f"/r/kernel/kernel_sleep_v2/--seed_{42 if r == 0 else 1000 * r}_--duration_40/rep001__idle", "sleep", "idle", r, 42 if r == 0 else 1000 * r, label="idle"))
    for m in range(1, n_members + 1):
        for r in range(reps):
            cells.append(_cell(f"/r/fam/fam_member_{m}/--seed_{42 if r == 0 else 1000 * r}_--duration_40/rep001__synth", f"fam_member_{m}", "unknown", r, 42 if r == 0 else 1000 * r,
                               status=schema.STATUS_UNKNOWN_KERNEL))
    return cells


def base_rows(n_members: int = 3) -> list[str]:
    letters = {1: "A", 2: "A", 3: "B"}
    out = [f"fam/fam_member_{m},sandbox,{m},{letters.get(m, 'C')},,,," for m in range(1, n_members + 1)]
    out += ["kernel,benign_kernel,,,,,,", "kernel/kernel_sleep_v2,idle,,,,,,"]
    return out


def write_classes(path: Path, lines: list[str], header: str = HDR) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([header] + lines) + "\n")
    return path


def write_cells_csv(out: Path, cells: list[dict]) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "cells.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(schema.CELLS_COLUMNS)); w.writeheader()
        for c in cells:
            w.writerow({k: c.get(k, "") for k in schema.CELLS_COLUMNS})
    return out / "cells.csv"


def validate(lines, cells=None, header=HDR, **kw) -> dict:
    o = tmp_out()
    p = write_classes(o / "classes.csv", lines, header)
    return C.validate_classes(p, cells if cells is not None else make_cells(), **kw)


# --------------------------------------------------------------------------- the validator

def test_validate_ok_and_counts():
    res = validate(base_rows())
    assert res["status"] == "ok" and res["refusals"] == []
    assert res["counts"] == {"sandbox": 6, "benign_kernel": 4, "idle": 2}
    assert res["members"] == {"1": "A", "2": "A", "3": "B"} and res["subfamilies"] == {"A": [1, 2], "B": [3]}
    assert res["unmatched_rows"] == 0 and res["unassigned_cells"] == 0
    assert any(w.startswith("campaign token defaulted to stage1") for w in res["warnings"])


def test_validate_campaign_label_silences_warning():
    res = validate(base_rows(), campaign_label="s1")
    assert res["status"] == "ok" and not res["warnings"]


@pytest.mark.parametrize("lines,header,expected", [
    (base_rows(), HDR + ",note", "refused: unknown column note"),
    (base_rows(), "path_prefix,class,member_index,subfamily_letter,rep,order_index,family", "refused: missing column workload_key"),
    (base_rows() + ["kernel/kernel_gemm_v2,mystery,,,,,,"], HDR, "refused: unknown class mystery in row 6"),
    (["fam/fam_member_1,sandbox,,A,,,,"] + base_rows()[1:], HDR, "refused: sandbox row 1 without member_index"),
    (["fam/fam_member_1,sandbox,1,,,,,"] + base_rows()[1:], HDR, "refused: sandbox row 1 without subfamily_letter"),
    (["fam/fam_member_1,sandbox,zero,A,,,,"] + base_rows()[1:], HDR, "refused: member_index not a positive integer in row 1"),
    (["fam/fam_member_1,sandbox,1,ab,,,,"] + base_rows()[1:], HDR, "refused: subfamily_letter not one capital letter in row 1"),
    (base_rows() + ["fam/fam_member_1/--seed_42_--duration_40/rep001__synth,sandbox,1,C,,,,"], HDR, "refused: member 1 mapped to two sub-families (A, C)"),
    (base_rows() + ["fam/fam_member_1/--seed_42_--duration_40/rep001__synth,sandbox,2,A,,7,,"], HDR, "refused: rows 6 and 1 disagree on member_index for one cell"),
    (base_rows() + ["kernel,benign_kernel,,,,,,"], HDR, "refused: duplicate path_prefix in rows 4 and 6"),
    (base_rows() + ["kernel/../fam,benign_kernel,,,,,,"], HDR, 'refused: path_prefix contains ".." in row 6'),
    (base_rows() + ["fam/fam_member_1/--seed_42_--duration_40/rep001__synth,sandbox,1,A,9,,,"], HDR, "refused: rep outside 0..7 in row 6"),
    (base_rows() + ["fam/fam_member_1/--seed_42_--duration_40/rep001__synth,sandbox,1,A,,0,,"], HDR, "refused: order_index not a positive integer in row 6"),
    (base_rows() + ["fam/fam_member_1/--seed_42_--duration_40/rep001__synth,sandbox,1,A,,3,,", "fam/fam_member_2/--seed_42_--duration_40/rep001__synth,sandbox,2,A,,3,,"], HDR, "refused: duplicate order_index 3"),
    (["fam/fam_member_1,sandbox,1,A,0,,,"] + base_rows()[1:], HDR, "refused: per-cell column rep on a row that matches 2 cells (row 1)"),
    (base_rows() + ["kernel/kernel_gemm_v2,benign_breadth,,,,,,"], HDR, "refused: benign_breadth row 6 without family"),
    (base_rows() + ["kernel/kernel_gemm_v2,benign_relaunched,,,,,,"], HDR, "refused: benign_relaunched row 6 without workload_key (parent kernel; CR3 2.31)"),
    (base_rows() + ["kernel/kernel_gemm_v2,benign_relaunched,,,,,,nokernel"], HDR, "refused: workload_key nokernel of benign_relaunched row 6 is not a kernel name"),
    (["kernel/kernel_sleep_v2,benign_kernel,,,,,,", "kernel,benign_kernel,,,,,,"] + base_rows()[:3], HDR, "refused: benign_kernel row 1 matches a cell whose role is idle"),
    (base_rows() + ["fam/fam_member_1/,sandbox,4,C,,,,"], HDR, "refused: two members of one workload path in rows 1 and 6"),
    (base_rows() + ["kernel/kernel_gemm_v2,external,1,,,,,"], HDR, "refused: member_index 1 used by both sandbox and external rows"),
])
def test_validate_refusals(lines, header, expected):
    res = validate(lines, header=header)
    assert res["status"] == "refused", res
    assert expected in res["refusals"], res["refusals"]


def test_validate_relaunched_own_grouping_needs_no_workload_key():
    res = validate(base_rows() + ["kernel/kernel_gemm_v2,benign_relaunched,,,,,,"], relaunched_grouping="own")
    assert res["status"] == "ok"


def test_validate_unmatched_row_is_a_warning_not_a_refusal():
    res = validate(base_rows() + ["fam/fam_member_9,sandbox,9,C,,,,"])
    assert res["status"] == "ok" and res["unmatched_rows"] == 1


def test_validate_missing_file_exit_2():
    o = tmp_out()
    write_cells_csv(o, make_cells())
    r = run_cli("classes", "validate", "--out", o, "--classes", o / "nowhere.csv", check=False)
    assert r.returncode == 2 and "missing input" in r.stderr


# --------------------------------------------------------------------------- apply, the join, the letter sequence

def _apply_out(lines=None, campaign_label=None) -> Path:
    o = tmp_out()
    write_cells_csv(o, make_cells())
    write_classes(o / "inputs" / "classes.csv", lines or base_rows())
    res = C.apply_classes(o, campaign_label=campaign_label)
    assert res["status"] == "ok" and res["applied"], res
    return o


def test_apply_rewrites_public_ids_and_is_idempotent():
    o = _apply_out()
    cells = {r["cell_id"]: r for r in rows(o / "cells.csv")}
    assert "sandbox_member_3__rep01__stage1" in cells
    c = cells["sandbox_member_3__rep01__stage1"]
    assert c["role"] == "sandbox" and c["status"] == "ok" and c["kernel"] == "sandbox_member_3" and c["archetype_predicted"] == "sandbox" and c["campaign"] == "stage1"
    assert "idle__rep00__idle" in cells and cells["idle__rep00__idle"]["kernel"] == "idle"
    assert not any("fam_member" in cid for cid in cells)
    assert (o / "cells.pre_classes.csv").is_file()
    pre = (o / "cells.pre_classes.csv").read_bytes()
    before = (o / "cells.csv").read_bytes()
    res2 = C.apply_classes(o)
    assert res2["status"] == "ok" and (o / "cells.csv").read_bytes() == before and (o / "cells.pre_classes.csv").read_bytes() == pre
    idx = jload(o / "cells.index.json")
    assert idx["classes_applied"]["counts"]["sandbox"] == 6
    join = rows(C.join_path(o))
    assert {j["class"] for j in join} == {"sandbox", "benign_kernel", "idle"}
    j = next(j for j in join if j["cell_id"] == "sandbox_member_3__rep01__stage1")
    assert j["y"] == "sandbox" and j["workload_key"] == "sandbox_member_3" and j["subfamily_letter"] == "B" and j["order_token"] == "S3r1" and j["split_role"] == "train_test"
    jk = next(j for j in join if j["class"] == "benign_kernel")
    assert jk["family"] == "kernels" and jk["order_token"].startswith("Br")
    meta = jload(C.det_dir(o) / "cell_classes.json")
    assert meta["S"] == 3 and meta["B"] == 3 and meta["n_assignments"] == 20


def test_apply_campaign_label_sets_the_token():
    o = _apply_out(campaign_label="c9")
    assert any(r["cell_id"].endswith("__c9") and r["role"] == "sandbox" for r in rows(o / "cells.csv"))


def test_apply_refusing_file_leaves_cells_untouched():
    o = tmp_out()
    write_cells_csv(o, make_cells())
    before = (o / "cells.csv").read_bytes()
    write_classes(o / "inputs" / "classes.csv", base_rows() + ["kernel,benign_kernel,,,,,,"])
    res = C.apply_classes(o)
    assert res["status"] == "refused" and not res["applied"]
    assert (o / "cells.csv").read_bytes() == before and not (o / "cells.pre_classes.csv").exists()
    assert jload(C.det_dir(o) / "classes_validation.json")["status"] == "refused"


def test_apply_refuses_stray_extract_dirs():
    o = tmp_out()
    write_cells_csv(o, make_cells())
    (o / "extract" / "fam_member_1__rep00__synth").mkdir(parents=True)
    write_classes(o / "inputs" / "classes.csv", base_rows())
    res = C.apply_classes(o)
    assert res["status"] == "refused" and res["refusals"] == [V.refused("extract/ holds 1 directories not keyed by a public cell_id; remove them and re-run")]


def test_apply_copies_classes_and_refuses_a_different_existing_file():
    o = tmp_out()
    write_cells_csv(o, make_cells())
    src = write_classes(o / "elsewhere.csv", base_rows())
    assert C.apply_classes(o, src)["status"] == "ok" and (o / "inputs" / "classes.csv").is_file()
    other = write_classes(o / "other.csv", base_rows()[:-1])
    res = C.apply_classes(o, other)
    assert res["status"] == "refused" and "inputs/classes.csv exists" in res["refusals"][0]


def test_join_archetype_rule_and_unassigned():
    o = tmp_out()
    write_cells_csv(o, make_cells())
    write_classes(o / "inputs" / "classes.csv", base_rows()[1:])        # no row for member 1: its cells are unassigned
    res = C.apply_classes(o, kernel_family_rule="archetype")
    assert res["status"] == "ok" and res["unassigned_cells"] == 2
    cells = rows(o / "cells.csv")
    assert sum(1 for c in cells if c["status"] == schema.STATUS_UNKNOWN_KERNEL) == 2     # untouched, still refused by plan11
    join = rows(C.join_path(o))
    assert {j["family"] for j in join if j["class"] == "benign_kernel"} == {"WORKING-SET"}
    assert not any(j["class"] == C.UNASSIGNED for j in join)      # a refused cell is not an ok cell; the join lists ok cells
    # an ok cell without a class row is listed as unassigned with every derived column empty
    c1 = next(c for c in cells if c["kernel"] == "fam_member_1")
    c1["status"] = "ok"
    write_cells_csv(o, cells)
    join = C.build_join(o, kernel_family_rule="archetype")
    un = [j for j in join if j["class"] == C.UNASSIGNED]
    assert len(un) == 1 and un[0]["y"] == "" and un[0]["workload_key"] == ""
    assert len(C.load_join(o)) == len(join) - 1 and len(C.load_join(o, include_unassigned=True)) == len(join)


def test_letter_sequence_and_not_run():
    o = _apply_out()
    toks = C.letter_sequence(C.load_join(o))
    assert toks == [V.not_run("order_index missing for 12 cells")]
    assert (C.det_dir(o) / "letter_sequence.txt").read_text().strip() == toks[0]
    lines = base_rows()
    cells = rows(o / "cells.pre_classes.csv")
    for i, c in enumerate(cells):
        t4 = "/".join(Path(c["path"]).parts[-4:])
        cls = {"kernel": "benign_kernel", "idle": "idle", "unknown": "sandbox"}[c["role"]]
        m = c["kernel"].split("_")[-1] if cls == "sandbox" else ""
        letter = {"1": "A", "2": "A", "3": "B"}.get(m, "")
        lines.append(f"{t4},{cls},{m},{letter},,{i + 1},,")
    write_classes(o / "inputs" / "classes.csv", lines)
    assert C.apply_classes(o)["status"] == "ok"
    toks = C.letter_sequence(C.load_join(o))
    assert toks == ["Br0", "Br1", "Br0", "Br1", "Ir0", "Ir1", "S1r0", "S1r1", "S2r0", "S2r1", "S3r0", "S3r1"]
    seq = rows(C.det_dir(o) / "letter_sequence.csv")
    assert [r["token"] for r in seq] == toks and seq[0]["order_index"] == "1"


def test_order_token_grammar():
    assert C.order_token("sandbox", 3, 0) == "S3r0" and C.order_token("benign_kernel", 0, 5) == "Br5" and C.order_token("idle", 0, 2) == "Ir2"
    assert C.order_token("harness_idle", 0, 0) == "Hr0" and C.order_token("benign_relaunched", 0, 7) == "Rr7" and C.order_token("external", 1, 0) == "X1r0"


# --------------------------------------------------------------------------- inherit-selection

def test_inherit_selection_default_and_from_and_refusal():
    o = tmp_out(); (o / "inputs").mkdir()
    p = C.inherit_selection(o, default_grid="W8_H4")
    sel = jload(p)
    assert set(S.RUNGS) <= set(sel) and sel["apf"]["grid_id"] == "W8_H4" and sel["params"]["grid_source"] == "default: W8_H4 (no inherited selection)"
    assert jload(C.det_dir(o) / "inherit_selection.json")["status"] == "ok"
    assert S.selected_grid_id(o, "content", None)[0] == "W8_H4" and C.grid_source(o).startswith("default:")
    src = o / "enc_selection.json"
    src.write_text(json.dumps({"schema": "plan11.selection.v1", "params": {"x": 1}, "citation": "c", "apf": {"grid_id": "W16_H8", "W": 16, "H": 8}}))
    C.inherit_selection(o, from_path=src, force=True)
    sel = jload(o / "gates" / "selection.json")
    assert sel["apf"]["grid_id"] == "W16_H8" and sel["params"]["grid_source"] == f"inherited: {src}" and sel["params"]["inherited_sha256"] == S.sha256_file(src)
    # a file written by gates_temporal select (no params.grid_source) is refused without --force
    (o / "gates" / "selection.json").write_text(json.dumps({"schema": "plan11.selection.v1", "params": {}, "citation": "c", "apf": {"grid_id": "W8_H2"}}))
    rec = C.inherit_selection(o, default_grid="W8_H4")
    assert rec.name == "inherit_selection.json"
    assert jload(rec)["status"] == V.refused("gates/selection.json exists and was written by gates_temporal select; pass --force to replace")
    assert jload(o / "gates" / "selection.json")["apf"]["grid_id"] == "W8_H2"
    with pytest.raises(ValueError):
        C.inherit_selection(o)


def test_extend_head_drop_adds_member_rows():
    o = _apply_out()
    S.write_head_drop_template(o / "inputs" / "head_drop.csv")
    C.extend_head_drop(o, C.load_join(o))
    hd = S.load_head_drop(o / "inputs" / "head_drop.csv")
    assert hd["sandbox_member_1"] == 0 and hd["gemm"] == 0 and hd["idle"] == 0
    assert C.apply_classes(o)["status"] == "ok"
    assert sum(1 for r in rows(o / "inputs" / "head_drop.csv") if r["kernel"] == "sandbox_member_1") == 1
