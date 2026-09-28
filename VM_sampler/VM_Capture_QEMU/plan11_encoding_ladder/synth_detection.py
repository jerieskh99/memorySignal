#!/usr/bin/env python3
"""synth_detection.py -- the synthetic two-class corpus with known answers (SPEC_DETECTION.md
section 4, builder A), built on synth.py. Every test of the detection layer uses it; there is no
real data on the build machine.

The benign side is the encoding corpus of SPEC 5.2 (12 kernels x reps at the presets, `idle`
idle cells) built by ``synth.corpus_specs(cv_case="preset", ...)``. The sandbox side is eight
members under the family directory ``synthfam`` with the test label ``synthfam_member_<m>``, so
``schema.parse_cell_path`` yields role ``unknown`` and the index lists the cells as
``refused: unknown kernel`` until ``classes apply`` runs (the same path the real cells take).

  synth_detection.py corpus --root R [--n-pairs 240] [--reps 8] [--idle 8] [--kernels all|<list>]
      [--members all|<list>] [--seed 20260916] [--order-confound on|off] [--drift-level]
      [--campaign-labels one|round_robin|confounded] [--idle-campaigns 1|2] [--write-classes]
      [--write-boundaries] [--stage2-fixture] [--conjunction-members] [--sandbox-n-pairs N]
      [--member8-k0 40000] [--drift-rate 0.002] [--no-compress] [--jobs 1]

Corrections from the reviews applied here: al-Kindi 2.8 (member 8's shape is spmm's, so its known
answer is ``nearest_family = kernels``); ML 2.4 (``--conjunction-members``, the empty-quarantine
case); ML 2.6 (``--sandbox-n-pairs``, the cadence-leak case). Deviation noted in the build report:
the harness-idle fixture's test label is ``harness_floor`` (SPEC 4.3 says ``harness_idle``), because
``schema.IDLE_MARKERS_DEFAULT`` would mark a label containing ``idle`` as the idle role and its id
would collide with the idle cells'; and ``--member8-k0`` exists because a 40,000-page cell costs
minutes to synthesize (the SPEC's 40,000 stays the default).

No server path appears here. No sandbox workload is named here.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plan11_encoding_ladder import schema, synth  # noqa: E402
from plan11_encoding_ladder.synth import SynthSpec, rep_seed  # noqa: E402

CITATION = "SPEC_DETECTION.md section 4 (the synthetic two-class corpus with known answers); SPEC section 5 (synth.py)"
SANDBOX_FAMILY_DIR = "synthfam"
SANDBOX_LABEL_FMT = "synthfam_member_{name}"
MEMBER_LETTERS = {1: "A", 2: "A", 3: "A", 4: "A", 5: "B", 6: "C", 7: "C", 8: "C"}
MEMBER8_K0 = 40000
CAMPAIGN_LABELS_RR = ("sandbox_deepdive_01c", "sandbox_deepdive_01c1", "dwarfs1_synth")
CONFOUNDED_LABEL_OF_ARCHETYPE = {"WORKING-SET": "sandbox_deepdive_01c", "SCATTER": "sandbox_deepdive_01c1",
                                 "SEQUENTIAL-GROW": "dwarfs1_synth", "FRONTIER-CHURN": "dwarfs1_synth"}
CONFOUNDED_K_NOISE = {"sandbox_deepdive_01c": 0.02, "sandbox_deepdive_01c1": 0.05, "dwarfs1_synth": 0.08}
IDLE_CAMPAIGN_LABELS = ("sandbox_deepdive_01c", "dwarfs1_synth")
IDLE_CAMPAIGN_FLOOR_F = (150, 190)
IDLE_CAMPAIGN_FLOOR_CHURN = (0.02, 0.10)   # a shape difference too: a floor-level shift alone is invisible to level-normalized features
DRIFT_RATE = 0.002                          # SPEC_DETECTION 4.3: K0 times (1 + DRIFT_RATE * order_rank) under --drift-level
STAGE2_RELAUNCH_FAMILY = "relaunch"
STAGE2_RELAUNCH_FMT = "relaunched_{name}"
STAGE2_HARNESS_FAMILY = "harnessidle"
STAGE2_HARNESS_FMT = "harness_{name}"
STAGE2_HARNESS_NAME = "floor"          # not "idle": schema.IDLE_MARKERS_DEFAULT would make it the idle role
STAGE2_HARNESS_FLOOR_F = 300
STAGE2_N_CELLS = 4
CLASSES_COLUMNS = ("path_prefix", "class", "member_index", "subfamily_letter", "rep", "order_index", "family", "workload_key")


def member_preset(m: int, *, member8_k0: int = MEMBER8_K0, conjunction: bool = False) -> dict:
    """The preset of member ``m`` (SPEC_DETECTION 4.2 table). Members 1 to 4 (sub-family A): spin
    content at a benign level, separable on the amount axis; member 5 (B): double content with churn
    0.60, separable on the identity axis only; member 6 (C): the idle preset, at floor; member 7 (C):
    the spmm preset exactly; member 8 (C): a level no benign workload reaches. With ``conjunction``
    members 1 to 4 take spin bytes (200, 300), so that ``r_l0`` alone sits between gibbs's and the
    double kernels' and ``r_l1l0 = 1`` alone equals gibbs's: only the pair separates (ML 2.4)."""
    if m in (1, 2, 3, 4):
        k0 = {1: 2048, 2: 4096, 3: 1024, 4: 3072}[m]
        return dict(K0=k0, content="spin", spin_bytes=((200, 300) if conjunction else (48, 96)), churn=0.02, k_noise=0.02)
    if m == 5:
        return dict(K0=2048, content="double", churn=0.60, k_noise=0.02)
    if m == 6:
        return dict(K0=0, floor_F=150, floor_churn=0.02, content="idle")
    if m == 7:
        p = dict(synth.PRESETS["spmm"])
        return p
    if m == 8:
        return dict(K0=int(member8_k0), content="double", churn=0.05, k_noise=0.02)
    raise ValueError(f"member index must be 1..8, got {m}")


def _parse_members(s: str | None) -> list[int]:
    if not s or s == "all":
        return list(range(1, 9))
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        m = int(x)
        if m not in MEMBER_LETTERS:
            raise ValueError(f"member index must be 1..8, got {m}")
        out.append(m)
    return out


def _parse_kernels(s: str | None) -> list[str]:
    if not s or s == "all":
        return list(schema.KERNEL_NAMES)
    ks = [x.strip() for x in s.split(",") if x.strip()]
    for k in ks:
        if k not in schema.KERNEL_NAMES:
            raise ValueError(f"unknown kernel {k}")
    return ks


def _interleaved(n_workloads: int, reps: int) -> list[int]:
    """Positions ``reps * j + r`` listed even first, then odd (SPEC_DETECTION 4.3, ``--order-confound off``):
    returns the position of each (j, r) in the realized order, indexed by ``reps * j + r``."""
    n = n_workloads * reps
    seq = [p for p in range(n) if p % 2 == 0] + [p for p in range(n) if p % 2 == 1]
    rank = [0] * n
    for i, p in enumerate(seq):
        rank[p] = i
    return rank


def corpus_specs(*, n_pairs: int = 240, reps: int = 8, idle: int = 8, kernels=None, members=None, seed: int = 20260916,
                 order_confound: str = "off", drift_level: bool = False, campaign_labels: str = "one",
                 idle_campaigns: int = 1, stage2_fixture: bool = False, conjunction_members: bool = False,
                 sandbox_n_pairs: int | None = None, member8_k0: int = MEMBER8_K0, drift_rate: float = DRIFT_RATE) -> tuple[list[dict], dict]:
    """The specs of the two-class corpus as (records, layout). Each record is {spec, class,
    member_index, subfamily_letter, order_index, rep, workload_key, family}. The realized order (the
    stage-1 shape): the kernel cells first, then the member cells, then the idle cells (then the
    stage-2 fixture cells); under ``order_confound = "off"`` the halves are interleaved inside every
    workload, under ``"on"`` the positions stay in blocks and every member's ``k_noise`` is
    ``0.02 + 0.03 (m - 1)``, the kernels' ``0.02 + 0.01 block``. ``drift_level`` multiplies every K0 by
    ``1 + drift_rate * order_rank`` (DRIFT_RATE = 0.002 as SPEC 4.3 says; the tests pass 0.02 so that the
    drift clears the jitter of a three-cell workload). With ``idle_campaigns = 2`` the second idle set
    carries floor_F 190 and floor_churn 0.10 against 150 and 0.02 (the SPEC names the level shift
    alone, which level normalization removes by construction). Citation: SPEC_DETECTION 4.2, 4.3."""
    if order_confound not in ("on", "off"):
        raise ValueError("order_confound must be on or off")
    if campaign_labels not in ("one", "round_robin", "confounded"):
        raise ValueError("campaign_labels must be one, round_robin or confounded")
    if int(idle_campaigns) not in (1, 2):
        raise ValueError("idle_campaigns must be 1 or 2")
    kernels = _parse_kernels(kernels) if not isinstance(kernels, (list, tuple)) else list(kernels)
    members = _parse_members(members) if not isinstance(members, (list, tuple)) else [int(m) for m in members]
    for m in members:
        if m not in MEMBER_LETTERS:
            raise ValueError(f"member index must be 1..8, got {m}")
    reps, idle, n_pairs = int(reps), int(idle), int(n_pairs)
    base = synth.corpus_specs(n_pairs=n_pairs, reps=reps, idle=idle, seed=seed, cv_case="preset")
    kern_specs = [s for s in base if s.name != synth.IDLE_PRESET["name"] and s.name in kernels]
    idle_specs = [s for s in base if s.name == synth.IDLE_PRESET["name"]]
    records: list[dict] = []
    # --- kernels ---
    kidx = {k: j for j, k in enumerate(kernels)}
    rank_k = _interleaved(len(kernels), reps) if order_confound == "off" else list(range(len(kernels) * reps))
    ki_of = {k: i for i, k in enumerate(schema.KERNEL_NAMES)}
    for s in kern_specs:
        j = kidx[s.name]
        r = next(rr for rr in range(reps) if rep_seed(rr, ki_of[s.name]) == s.seed)
        pos = rank_k[reps * j + r]
        if order_confound == "on":
            s.k_noise = 0.02 + 0.01 * j
        label = "synth"
        if campaign_labels == "round_robin":
            label = CAMPAIGN_LABELS_RR[r % len(CAMPAIGN_LABELS_RR)]
        elif campaign_labels == "confounded":
            label = CONFOUNDED_LABEL_OF_ARCHETYPE[schema.ARCHETYPE_OF[s.name]]
            s.k_noise = CONFOUNDED_K_NOISE[label]
        s.label = label
        records.append({"spec": s, "class": "benign_kernel", "member_index": 0, "subfamily_letter": "-",
                        "order_index": pos + 1, "rep": r, "workload_key": s.name, "family": "kernels"})
    off = len(kernels) * reps
    # --- members ---
    rank_m = _interleaved(len(members), reps) if order_confound == "off" else list(range(len(members) * reps))
    for jm, m in enumerate(members):
        p = member_preset(m, member8_k0=member8_k0, conjunction=conjunction_members)
        for r in range(reps):
            s = SynthSpec(name=str(m), seed=rep_seed(r, 12 + m), n_pairs=int(sandbox_n_pairs or n_pairs), rep_dir=1, label="synth",
                          family=SANDBOX_FAMILY_DIR, test_label_fmt=SANDBOX_LABEL_FMT, **p)
            if order_confound == "on":
                s.k_noise = 0.02 + 0.03 * (m - 1)
            pos = off + rank_m[reps * jm + r]
            records.append({"spec": s, "class": "sandbox", "member_index": m, "subfamily_letter": MEMBER_LETTERS[m],
                            "order_index": pos + 1, "rep": r, "workload_key": f"sandbox_member_{m}", "family": "sandbox"})
    off += len(members) * reps
    # --- idle ---
    rank_i = _interleaved(1, idle) if order_confound == "off" else list(range(idle))
    for r, s in enumerate(idle_specs):
        if int(idle_campaigns) == 2:
            half = 0 if r < idle // 2 else 1
            s.label = IDLE_CAMPAIGN_LABELS[half]
            s.floor_F = IDLE_CAMPAIGN_FLOOR_F[half]
            s.floor_churn = IDLE_CAMPAIGN_FLOOR_CHURN[half]
        pos = off + rank_i[r]
        records.append({"spec": s, "class": "idle", "member_index": 0, "subfamily_letter": "-",
                        "order_index": pos + 1, "rep": r, "workload_key": "idle", "family": "idle"})
    off += idle
    # --- stage 2 fixture ---
    if stage2_fixture:
        gp = dict(synth.PRESETS["gemm"])
        for r in range(STAGE2_N_CELLS):
            s = SynthSpec(name="gemm", seed=rep_seed(r, 30), n_pairs=n_pairs, rep_dir=1, label="synth",
                          family=STAGE2_RELAUNCH_FAMILY, test_label_fmt=STAGE2_RELAUNCH_FMT, **gp)
            records.append({"spec": s, "class": "benign_relaunched", "member_index": 0, "subfamily_letter": "-",
                            "order_index": off + r + 1, "rep": r, "workload_key": "gemm", "family": "relaunched"})
        off += STAGE2_N_CELLS
        hp = dict(synth.IDLE_PRESET); hp.pop("name"); hp.pop("label"); hp["floor_F"] = STAGE2_HARNESS_FLOOR_F
        for r in range(STAGE2_N_CELLS):
            s = SynthSpec(name=STAGE2_HARNESS_NAME, seed=rep_seed(r, 31), n_pairs=n_pairs, rep_dir=1, label="synth",
                          family=STAGE2_HARNESS_FAMILY, test_label_fmt=STAGE2_HARNESS_FMT, **hp)
            records.append({"spec": s, "class": "harness_idle", "member_index": 0, "subfamily_letter": "-",
                            "order_index": off + r + 1, "rep": r, "workload_key": "harness_idle", "family": "harness_idle"})
        off += STAGE2_N_CELLS
    if drift_level:
        for rec in records:
            rec["spec"].K0 = int(round(rec["spec"].K0 * (1.0 + float(drift_rate) * (rec["order_index"] - 1))))
    layout = {"n_pairs": n_pairs, "reps": reps, "idle": idle, "kernels": kernels, "members": members, "seed": seed,
              "order_confound": order_confound, "drift_level": drift_level, "campaign_labels": campaign_labels,
              "idle_campaigns": int(idle_campaigns), "stage2_fixture": stage2_fixture, "conjunction_members": conjunction_members,
              "sandbox_n_pairs": sandbox_n_pairs, "member8_k0": member8_k0, "drift_rate": drift_rate, "member_letters": {str(k): v for k, v in MEMBER_LETTERS.items()}}
    return records, layout


def _tail4(path: Path) -> str:
    parts = [p for p in Path(path).parts if p not in ("", "/")]
    return "/".join(parts[-4:])


def write_classes_csv(path: Path, records: list[dict], paths: list[str], *, with_order: bool = True) -> Path:
    """``<root>/classes.csv`` in the format of SPEC_DETECTION 2.1: the member rows (``synthfam/synthfam_member_<m>``),
    the kernel row (``kernel``), the idle row (the idle test label's prefix), the stage-2 rows when present,
    and (``with_order``) one per-cell row with ``order_index`` per cell, keyed by its last four path components."""
    rows: list[dict] = []
    members = sorted({r["member_index"] for r in records if r["class"] == "sandbox"})
    for m in members:
        rows.append({"path_prefix": f"{SANDBOX_FAMILY_DIR}/{SANDBOX_LABEL_FMT.format(name=m)}", "class": "sandbox",
                     "member_index": m, "subfamily_letter": MEMBER_LETTERS[m]})
    if any(r["class"] == "benign_kernel" for r in records):
        rows.append({"path_prefix": "kernel", "class": "benign_kernel"})
    if any(r["class"] == "idle" for r in records):
        idle_spec = next(r["spec"] for r in records if r["class"] == "idle")
        rows.append({"path_prefix": f"{idle_spec.family}/{idle_spec.test_label}", "class": "idle"})
    if any(r["class"] == "benign_relaunched" for r in records):
        rows.append({"path_prefix": f"{STAGE2_RELAUNCH_FAMILY}/{STAGE2_RELAUNCH_FMT.format(name='gemm')}", "class": "benign_relaunched",
                     "workload_key": "gemm"})
    if any(r["class"] == "harness_idle" for r in records):
        rows.append({"path_prefix": f"{STAGE2_HARNESS_FAMILY}/{STAGE2_HARNESS_FMT.format(name=STAGE2_HARNESS_NAME)}", "class": "harness_idle"})
    if with_order:
        for rec, p in zip(records, paths):
            row = {"path_prefix": _tail4(p), "class": rec["class"], "order_index": rec["order_index"]}
            if rec["class"] == "sandbox":
                row["member_index"] = rec["member_index"]; row["subfamily_letter"] = rec["subfamily_letter"]
            if rec["class"] == "benign_relaunched":
                row["workload_key"] = "gemm"
            rows.append(row)
    path = Path(path)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CLASSES_COLUMNS))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in CLASSES_COLUMNS})
    return path


def expected_cell_id(rec: dict, campaign_token: str = "stage1") -> str:
    """The public cell id ``classes apply`` will assign (SPEC_DETECTION 2.3): the campaign token of a
    sandbox cell is ``stage1`` unless ``--campaign-label`` says otherwise (al-Farabi M1)."""
    s, r = rec["spec"], rec["rep"]
    cls = rec["class"]
    if cls == "benign_kernel":
        return f"{s.name}__rep{r:02d}__{schema.campaign_of(s.label)}"
    if cls == "sandbox":
        return f"sandbox_member_{rec['member_index']}__rep{r:02d}__{campaign_token}"
    if cls == "idle":
        return f"idle__rep{r:02d}__{schema.campaign_of(s.label)}"
    if cls == "benign_relaunched":
        return f"relaunched_gemm__rep{r:02d}__{schema.campaign_of(s.label)}"
    if cls == "harness_idle":
        return f"harness_idle__rep{r:02d}__{schema.campaign_of(s.label)}"
    raise ValueError(cls)


def write_boundaries_csv(path: Path, records: list[dict], paths: list[str]) -> Path:
    """``<root>/iteration_boundaries.csv`` (cell_id, boundary_seqs as ';'-separated ascending seqs) from
    ``truth.json``'s ``boundaries`` for every cell that has a pulse (SPEC_DETECTION 4.3; CR3 2.29)."""
    rows = []
    for rec, p in zip(records, paths):
        t = json.loads((Path(p) / "truth.json").read_text())
        b = t.get("boundaries") or []
        if b:
            rows.append({"cell_id": expected_cell_id(rec), "boundary_seqs": ";".join(str(int(x)) for x in b)})
    path = Path(path)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cell_id", "boundary_seqs"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def _worker(args):
    spec, root, compress = args
    return str(synth.write_cell(spec, root, compress=compress, keep_sets=False, truth_sides=("t",)))


def write_corpus(root, *, compress: bool = True, jobs: int = 1, write_classes: bool = False, write_boundaries: bool = False, **kw) -> dict:
    """Write the two-class corpus under ``root`` with ``corpus_detection.json`` (params, citation, every
    cell's path, class, member index, letter, order index, expected public cell id). With
    ``write_classes`` also ``<root>/classes.csv`` (2.1 format) and ``<root>/classes.no_order.csv`` (the
    same without the per-cell rows); with ``write_boundaries`` also ``<root>/iteration_boundaries.csv``."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    records, layout = corpus_specs(**kw)
    tasks = [(r["spec"], str(root), compress) for r in records]
    if int(jobs) > 1:
        import multiprocessing as mp
        order = sorted(range(len(tasks)), key=lambda i: -(records[i]["spec"].K0 + records[i]["spec"].floor_F + records[i]["spec"].pulse_extra))
        paths = [None] * len(tasks)
        with mp.Pool(processes=int(jobs)) as pool:
            for i, path in zip(order, pool.imap(_worker, [tasks[i] for i in order], chunksize=1)):
                paths[i] = path
    else:
        paths = [_worker(t) for t in tasks]
    cells = [{"path": p, "class": r["class"], "member_index": r["member_index"], "subfamily_letter": r["subfamily_letter"],
              "order_index": r["order_index"], "rep": r["rep"], "workload_key": r["workload_key"], "family": r["family"],
              "expected_cell_id": expected_cell_id(r), "spec": asdict(r["spec"])} for r, p in zip(records, paths)]
    manifest = {"schema": "plan11.detection.synth_corpus.v1", "params": {**layout, "compress": bool(compress), "jobs": int(jobs)},
                "citation": CITATION, "n_cells": len(cells), "cells": cells,
                "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if write_classes:
        write_classes_csv(root / "classes.csv", records, paths, with_order=True)
        write_classes_csv(root / "classes.no_order.csv", records, paths, with_order=False)
        manifest["classes_csv"] = str(root / "classes.csv")
    if write_boundaries:
        write_boundaries_csv(root / "iteration_boundaries.csv", records, paths)
        manifest["iteration_boundaries_csv"] = str(root / "iteration_boundaries.csv")
    (root / "corpus_detection.json").write_text(json.dumps(manifest, indent=1, default=str))
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="synth_detection.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("corpus")
    k.add_argument("--root", required=True)
    k.add_argument("--n-pairs", type=int, default=240)
    k.add_argument("--reps", type=int, default=8)
    k.add_argument("--idle", type=int, default=8)
    k.add_argument("--kernels", default="all")
    k.add_argument("--members", default="all")
    k.add_argument("--seed", type=int, default=20260916)
    k.add_argument("--order-confound", choices=("on", "off"), default="off")
    k.add_argument("--drift-level", action="store_true")
    k.add_argument("--campaign-labels", choices=("one", "round_robin", "confounded"), default="one")
    k.add_argument("--idle-campaigns", type=int, choices=(1, 2), default=1)
    k.add_argument("--write-classes", action="store_true")
    k.add_argument("--write-boundaries", action="store_true")
    k.add_argument("--stage2-fixture", action="store_true")
    k.add_argument("--conjunction-members", action="store_true")
    k.add_argument("--sandbox-n-pairs", type=int, default=None)
    k.add_argument("--member8-k0", type=int, default=MEMBER8_K0)
    k.add_argument("--drift-rate", type=float, default=DRIFT_RATE)
    k.add_argument("--no-compress", action="store_true")
    k.add_argument("--jobs", type=int, default=1)
    return ap


def main(argv=None) -> int:
    ap = _build_parser()
    a = ap.parse_args(argv)
    try:
        m = write_corpus(a.root, compress=not a.no_compress, jobs=a.jobs, write_classes=a.write_classes, write_boundaries=a.write_boundaries,
                         n_pairs=a.n_pairs, reps=a.reps, idle=a.idle, kernels=a.kernels, members=a.members, seed=a.seed,
                         order_confound=a.order_confound, drift_level=a.drift_level, campaign_labels=a.campaign_labels,
                         idle_campaigns=a.idle_campaigns, stage2_fixture=a.stage2_fixture, conjunction_members=a.conjunction_members,
                         sandbox_n_pairs=a.sandbox_n_pairs, member8_k0=a.member8_k0, drift_rate=a.drift_rate)
        print(f"[synth_detection] wrote {m['n_cells']} cells under {a.root}")
        return 0
    except ValueError as exc:
        print(f"usage: {exc}", file=sys.stderr)
        return 3
    except FileNotFoundError as exc:
        print(f"missing input: {exc}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
