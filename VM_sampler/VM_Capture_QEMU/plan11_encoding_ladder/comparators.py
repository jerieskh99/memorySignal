#!/usr/bin/env python3
"""comparators.py -- the three exact-input comparators of paper 2 (build epoch 2, builder A):
Savoldi 2010, Dhodapkar and Smith 2003, Law 2010, each computed per cell from the extract (or,
for Law, from the trajectory in a second streaming pass), written as one feature vector per cell
at the whole-cell point ``Wall_Hall`` and submitted to the same split stage and the same
comparison gates as the ladder's rungs.

Citation: ``apf_paper/council/14_hunayn_exact_input_comparators.md`` (C14) candidates 1, 3, 2
(the exact definitions); ``P2_STRUCTURE.md`` section 0 row "Baseline" (all three named) and
section V ("The splits", "Models", 5.1 Plan 08, 5.2 G-L, G-DIM, G-M, G-X: every gate applied
here runs through ``models.run_split_stage`` and ``gates_comparison`` unchanged);
``P2E_STRUCTURE.md`` section 7 (Savoldi and Dhodapkar-Smith for EUSIPCO, Law held for IFIP);
``P2_AUTHOR_ANSWERS.md`` "Decisions of 2026-09-17" (the picks accepted; the Dhodapkar-Smith
threshold a declared sweep, every point kept, 0.04 the default); ``SPEC_epoch2.md`` Part 1
(the interfaces) and Part 4 (every choice the definitions leave open, exposed as a parameter);
``SPEC_epoch2_review_al_farabi.md`` sections 2 (a), (b) and 5 items 2, 4, 5 (the record follows
the value; a default off the grid is refused or appended and recorded; the memory test wraps
the page array; the G-M row carries the split's own ``not applicable:`` string).

What each row means (the two variants written for every comparator; Part 4 items 10 and 11):
the raw row is the method as published, level-inclusive by construction; the level-normalized
row applies the count-rung rule of P2 Sec. V G-L (i) (divide by the cell's own median K) to
Savoldi's and Law's count features, and divides Dhodapkar-Smith's counts by the pair count (the
delta is a Jaccard, level-free, so no level division exists). Savoldi normalized: the first
feature is the mean over the median (a number near one that carries only the skew of the
count distribution), the second the relative spread; the row tests whether skew and relative
spread alone carry the label. Both rows are computed; which one Table 7 prints is
``tables.TABLE7_VARIANT``.

Layout (SPEC_epoch2 Part 1.2 to 1.6): ``gates/comparators/{savoldi,dhodapkar,law}.csv`` and their
``.params.json``, the sweeps ``dhodapkar_sweep.csv`` and ``law_sweep.csv`` (every grid point
kept, the default marked, none selected against labels), the per-kernel summaries
``*_per_kernel.csv``, Law's ``law_cells.json`` and ``law_series/<cell_id>.npz``, the feature
files ``features/cmp_<name>/Wall_Hall_{raw,norm}.npz`` (the keys of ``series.build_features``),
the split stage under ``gates/splits/cmp_<name>/Wall_Hall/``, the comparison gates under
``gates/comparators/{gl,gdim,gm}.csv`` (G-X in ``gates/gx.csv``), and ``verdicts.csv``.

CLI (SPEC_epoch2 Part 1.6): ``comparators.py savoldi | dhodapkar | law | gates | all --out O ...``;
``--no-splits`` computes the statistics and the feature files only (under ``all`` the gates are
skipped with the split stage they read); exit 0 on success (a written refusal included), 2 when
``cells.csv`` or a named input is missing (its path on stderr) or a declared default is not a grid
point, 1 on an internal error.

No server path appears in this file. No sandbox workload is named.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import traceback
import weakref
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plan11_encoding_ladder import schema  # noqa: E402
from plan11_encoding_ladder import series as S  # noqa: E402
from plan11_encoding_ladder import verdicts as V  # noqa: E402
from plan11_encoding_ladder import models as M  # noqa: E402
from plan11_encoding_ladder import gates_comparison as GX  # noqa: E402
from plan11_encoding_ladder.extract import Refusal, open_text  # noqa: E402

N = schema.N_PAGES
GRID_ID = schema.COMPARATOR_GRID_ID              # "Wall_Hall": one vector per cell, by definition
GRID_ID_REASON = "one vector per cell: the whole-cell point by definition"
CMP_DIR = "comparators"                          # gates/comparators/

# --------------------------------------------------------------------------- citations (verbatim in params)
CIT_SAVOLDI = ("C14 cand. 1: Savoldi, Gubian, Echizen 2010, 'Uncertainty in Live Forensics', Advances in Digital "
               "Forensics VI, IFIP AICT 337, pp. 171-184, DOI 10.1007/978-3-642-15506-2_12: per consecutive pair the "
               "number of differing 4 KiB pages; U = mu_dmp +/- sigma_dmp, the sample mean and standard deviation of that "
               "count over the run (EXACT input: our per-pair K); P2 Sec. 0 Baseline; P2E Sec. 7 (the one-number baseline); "
               "P2 Sec. V 'The splits', 'Models' and 5.1 Plan 08 (B1-G1, B1-G6) through models.run_split_stage")
CIT_DHODAPKAR = ("C14 cand. 3: Dhodapkar and Smith 2003, 'Comparing Program Phase Detection Techniques', MICRO-36, pp. 217-227 "
                 "(and ISCA 2002): delta_{i,i-1} = (|W_i u W_{i-1}| - |W_i n W_{i-1}|) / |W_i u W_{i-1}| between consecutive "
                 "working sets, a phase change when delta exceeds a threshold, stability and average phase length; with W_i "
                 "our changed-page set the delta is one minus Jaccard identically (C14: 'Their delta is one minus Jaccard, "
                 "identically'); P2 Sec. 0 Baseline; P2E Sec. 7 (the named method on the overlap axis); AA 2026-09-17 "
                 "(a declared sweep, every point kept, 0.04 the default); C14 author (declare delta_th before the labels or "
                 "sweep it and show the sweep); P2 Sec. V through models.run_split_stage")
CIT_LAW = ("C14 cand. 2: Law et al. 2010, 'Identifying Volatile Data from Multiple Memory Dumps in Live Forensics', "
           "Advances in Digital Forensics VI, IFIP AICT 337, pp. 185-194, DOI 10.1007/978-3-642-15506-2_13: a page is "
           "dynamic in X consecutive dumps if X or more consecutive hashes differ, static if identical; an index of run "
           "lengths answers all X in one pass; in our rows 'dynamic in X' = a membership run of length X-1 in the changed "
           "sets, 'static in X' = a non-membership run of length X-1 (C14's exact reading); P2 Sec. 0 Baseline (Law and "
           "Savoldi as Table 7 rows for the IFIP version); P2E Sec. 7 (held for IFIP); P2 Sec. V through models.run_split_stage")
CIT_GATES = ("P2 Sec. V 5.2 G-L (i), G-DIM, G-M, G-X applied to the comparator rows exactly as to a rung (SPEC 3.7.4, 3.7.7, "
             "3.7.8, 3.7.6; CR 2.2 items 24, 32, 33, 26); SPEC_epoch2 Part 1.5 (which gates apply and which do not, with the "
             "printed string of each); SPEC_epoch2_review_al_farabi.md 5.5 (the G-M row carries the split's own not-applicable string)")
INTERVAL_NOTE = ("C14 cand. 1: 'Only the interval differs (500 ms vs one acquisition time)'; ours is one snapshot pair "
                 "(about 0.644 s guest spacing, SPEC 2.3), theirs one acquisition time")

# --------------------------------------------------------------------------- constants (every one recorded in params)
SAVOLDI_ROWS = "all_after_head_drop"        # Part 4 item 1: "all_after_head_drop" | "rung_series"
SAVOLDI_DDOF = 1                            # Part 4 item 2: sample SD (C14: "sample mean ... and standard deviation")
DHODAPKAR_GRID = (0.04, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)   # Part 4 item 3: the brief's nine points plus the author's declared default
DHODAPKAR_DEFAULT = 0.04                    # AA 2026-09-17; marked "declared default" everywhere it is printed
DHODAPKAR_BOUNDARY_RULE = "gt"              # Part 4 item 4: boundary when delta > delta_th ("gt") or delta >= delta_th ("ge")
DHODAPKAR_PHASE_LENGTH_RULE = "n_over_b_plus_1"   # Part 4 item 5: n_pairs / (n_boundaries + 1) | "interior"
LAW_X_GRID = (2, 4, 8, 16)                  # the declared grid, every point computed and kept
LAW_X_DEFAULT = 4                           # Part 4 item 6
LAW_X_UNIT = "dumps"                        # Part 4 item 7: "dumps" (run length X-1, C14's reading) | "pairs" (run length X)
LAW_HEAD_DROP_RULE = "runs_from_seq_first"  # Part 4 item 8: "runs_from_seq_first" | "reset_at_head_drop"
LAW_FEATURE_SOURCE = "window"               # Part 4 item 9: "window" (the per-pair series) | "ever" (distinct pages over the cell)
COMPARATOR_NORM_RULE = "median_K"           # Part 4 item 10: the count-rung rule of P2 Sec. V G-L (i) applied to the comparator features
OFF_GRID_DEFAULT_RULE = "refuse"            # al-Farabi review 2 (b): a default that is not a grid point is refused (exit 2) or appended and recorded
GRID_TOL = 1e-9                             # float equality when a grid point is matched against the default

assert DHODAPKAR_DEFAULT == schema.COMPARATOR_DECLARED_DEFAULTS["cmp_dhodapkar"][1]
assert LAW_X_DEFAULT == schema.COMPARATOR_DECLARED_DEFAULTS["cmp_law"][1]

DHODAPKAR_DEFAULT_SOURCE = "AA 2026-09-17: 0.04 marked as the default"
DHODAPKAR_GRID_SOURCE = "SPEC_epoch2 Part 1.3: the brief's nine points 0.1 .. 0.9 plus the author's declared default (module constant DHODAPKAR_GRID)"
LAW_X_DEFAULT_SOURCE = "SPEC_epoch2 Part 4 item 6 (the author has declared no X)"
LAW_X_GRID_SOURCE = "SPEC_epoch2 Part 1.4: the declared grid (module constant LAW_X_GRID)"

SPLIT_PLAN = (("within_trace", "kernel"), ("within_trace", "archetype"), ("loro", "kernel"),
              ("loro", "archetype"), ("loko", "archetype"))
GM_SPLITS = (("loko", "archetype"), ("loro", "kernel"), ("within_trace", "kernel"))
PRIMARY_LABELSPACE = {"loko": "archetype", "loro": "kernel", "within_trace": "kernel"}

# The strings Table 7 prints for the gates that do not apply to a comparator row (SPEC_epoch2 Part 1.5)
NA_GC = V.not_applicable("comparator, not a lead of the ladder (G-C calibrates the ladder's leads against the gemm pulse)")
NA_GF1 = V.not_applicable("one row per cell (G-F (i) is the within-trace window design)")
NA_GP = V.not_applicable("per-kernel pass-period reading, not a row property")
NA_GDEC = V.not_applicable("a reading of the content rung on one kernel")
NA_TEMPORAL = V.not_applicable("no (W, H) (one vector per cell by definition)")
RESOLUTION_TEXT = "whole cell (per cell, by definition)"
GL_RAW_TEXT = "level-inclusive (as published)"
GL_PART_II = V.not_applicable("G-L (ii) is APF's shot-noise route")

# Instrumentation of the Law pass (SPEC_epoch2 Part 1.10 test 4; al-Farabi review 5.2): the hook is
# called once per finished snapshot with the snapshot's page wrapper; every live wrapper is in the set.
LAW_STEP_HOOK = None
_LIVE_PAGE_ARRAYS: "weakref.WeakSet[_PageList]" = weakref.WeakSet()

SAVOLDI_COLUMNS = ("cell_id", "kernel", "role", "archetype_predicted", "rep", "campaign", "admissible", "head_drop",
                   "rows_rule", "n_rows_used", "K_mean", "K_sd", "K_median", "U_mean_pct", "U_sd_pct", "U_text", "status")
SAVOLDI_KERNEL_COLUMNS = ("kernel", "n_cells", "K_mean_median", "K_mean_min", "K_mean_max", "K_sd_median", "U_text_median")
DHODAPKAR_SWEEP_COLUMNS = ("cell_id", "kernel", "role", "rep", "campaign", "admissible", "excluded_pair_rung", "delta_th",
                           "is_default", "n_pairs_used", "n_pairs_blank", "n_boundaries", "stability", "mean_phase_length_pairs")
DHODAPKAR_COLUMNS = ("cell_id", "kernel", "role", "archetype_predicted", "rep", "campaign", "admissible", "excluded_pair_rung",
                     "head_drop", "n_pairs_used", "n_pairs_blank", "delta_mean", "delta_q05", "delta_q25", "delta_q50",
                     "delta_q75", "delta_q95", "delta_th_default", "n_boundaries", "stability", "mean_phase_length_pairs", "status")
DHODAPKAR_KERNEL_COLUMNS = ("kernel", "n_cells", "n_boundaries_median", "stability_median", "mean_phase_length_median", "delta_q50_median")
LAW_SWEEP_COLUMNS = ("cell_id", "kernel", "role", "rep", "campaign", "admissible", "excluded_pair_rung", "X", "run_length_L",
                     "is_default", "n_windows", "dyn_mean", "dyn_sd", "dyn_min", "dyn_max", "dyn_frac_mean", "sta_mean", "sta_sd",
                     "sta_min", "sta_max", "sta_frac_mean", "dyn_ever", "sta_ever", "dyn_ever_frac", "sta_ever_frac", "status")
LAW_COLUMNS = ("cell_id", "kernel", "role", "archetype_predicted", "rep", "campaign", "admissible", "excluded_pair_rung",
               "head_drop", "X_default", "run_length_L", "n_windows", "dyn_mean", "dyn_sd", "sta_mean", "sta_sd", "dyn_ever",
               "sta_ever", "check_x2_equals_K", "status")
LAW_KERNEL_COLUMNS = ("kernel", "n_cells", "dyn_mean_median", "dyn_frac_mean_median", "sta_frac_mean_median", "dyn_ever_median")
GDIM_COLUMNS = ("rung", "grid_id", "d", "d_matched", "method", "status", "loko_score", "matched_to")
GM_COLUMNS = ("split", "rung_a", "rung_b", "score_a", "score_b", "diff", "spread", "improving", "worsening", "ties", "verdict")
VERDICT_COLUMNS = ("rung", "variant", "split", "labelspace", "feature_count", "accuracy", "macro_recall", "null_p95", "rank",
                   "majority", "b1_g1", "gl", "gdim", "gm_vs_apf", "gx", "status")


# =============================================================================================
# shared helpers
# =============================================================================================
def cmp_dir(out: Path) -> Path:
    return Path(out) / "gates" / CMP_DIR


def _pct_text(x: float, sig: int) -> str:
    """A percentage with ``sig`` significant digits in positional form (never an exponent), the
    per-run form of C14 ("U = 65.5% +/- 0.15%")."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "nan"
    return np.format_float_positional(float(x), precision=sig, unique=False, fractional=False, trim="-")


def _admissibility(out: Path) -> tuple[dict, set, bool, bool]:
    """(``{cell_id: "true" | "false"}`` from gates/preconditions.csv ``all_hard_pass``, the set
    ``excluded_cells_pair_rungs`` of gates/preconditions.json, csv present, json present)."""
    out = Path(out)
    pm = S.preconditions_map(out)
    adm = {cid: ("true" if str(r.get("all_hard_pass", "true")).lower() == "true" else "false") for cid, r in pm.items()}
    pj = out / "gates" / "preconditions.json"
    ex_pair: set = set()
    if pj.is_file():
        try:
            ex_pair = set(S.read_json(pj).get("excluded_cells_pair_rungs") or [])
        except (OSError, ValueError):
            ex_pair = set()
    return adm, ex_pair, bool(pm), pj.is_file()


def _cells_with_extract(out: Path) -> list[dict]:
    """Every ``status == ok`` cell of ``<out>/cells.csv`` whose ``extract/<cell_id>/extract.csv``
    exists, in cells.csv order (SPEC_epoch2 Part 1.1: the comparator does not re-decide
    admissibility; the split stage applies it at read time)."""
    out = Path(out)
    p = out / "cells.csv"
    if not p.is_file():
        raise FileNotFoundError(str(p))
    return [c for c in S.load_cells(p) if S.extract_path(out, c["cell_id"]).is_file()]


def _identity(c: dict) -> dict:
    return {"cell_id": c["cell_id"], "kernel": c["kernel"], "role": c["role"],
            "archetype_predicted": c.get("archetype_predicted", ""), "rep": int(c["rep"]), "campaign": c["campaign"]}


def _head_drops(out: Path, cells: list[dict]) -> tuple[dict, dict]:
    """(``{cell_id: head_drop}``, the head-drop table): the per-cell value is the kernel's entry,
    or the ``idle`` entry for an idle cell (SPEC 3.1.2; SPEC_epoch2 Part 1.1)."""
    hd = S.load_head_drop(Path(out) / "inputs" / "head_drop.csv")
    return {c["cell_id"]: S.head_drop_for(hd, "idle" if c["role"] == "idle" else c["kernel"]) for c in cells}, hd


def _kernel_label(c: dict) -> tuple[str, str]:
    """(kernel, archetype) as the feature file stores them: idle rows ``idle`` / ``IDLE`` (SPEC
    3.1.5), kernel rows the label-derived name and ``archetype_predicted`` of cells.csv (G-K0's
    relabelling is applied at read time by ``models.prepare_split_data``, never stored)."""
    if c["role"] == "idle":
        return "idle", "IDLE"
    return c["kernel"], c.get("archetype_predicted") or "unknown"


def _admissible_for_summary(adm_text: str) -> bool:
    """Per-kernel summaries are over admissible cells: ``all_hard_pass`` true, or every cell when
    the preconditions have not run (the split stage's own reading of a missing preconditions.csv)."""
    return adm_text != "false"


def _median_or_nan(vals) -> float:
    vals = [float(v) for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.median(vals)) if vals else float("nan")


def _kernel_groups(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    """Rows grouped by kernel, ``schema.KERNELS`` order then any other kernel, then ``idle``; a
    row's kernel is ``idle`` for an idle cell."""
    groups: dict = {}
    for r in rows:
        k = "idle" if r["role"] == "idle" else r["kernel"]
        groups.setdefault(k, []).append(r)
    order = [k for k in schema.KERNEL_NAMES if k in groups]
    order += sorted(k for k in groups if k not in schema.KERNEL_NAMES and k != "idle")
    if "idle" in groups:
        order.append("idle")
    return [(k, groups[k]) for k in order]


def write_feature_files(out: Path, name: str, cells: list[dict], vectors: dict, *, names_raw, names_norm,
                        n_series: dict, head_drop: dict, n_dropped: int, extra_params: dict | None = None) -> tuple[Path, Path]:
    """``features/<name>/Wall_Hall_raw.npz`` and ``Wall_Hall_norm.npz`` with the keys of
    ``series.build_features`` (SPEC 3.1.5; SPEC_epoch2 Part 1.6), so that ``series.load_features``,
    ``splits.make_labels``, ``models.prepare_split_data`` and ``models.run_split_stage`` read them
    unchanged: ``X`` [n_cells, d], ``feature_names``, per row ``cell_id, kernel, archetype, campaign,
    role, rep, win_start (= 0), n_series_cell``; scalars ``W = -1, H = -1, grid_id = "Wall_Hall",
    normalized, head_drop_json, n_windows_dropped`` (the cells with a refused status, omitted),
    ``wapf_norm = ""``. Rows: cells in cells.csv order; idle rows ``archetype = "IDLE"``,
    ``kernel = "idle"``. ``vectors`` = {cell_id: (raw_vector, norm_vector)}; a cell absent from it
    is omitted."""
    out = Path(out)
    paths = []
    hd_json = json.dumps(head_drop, sort_keys=True)
    for normalized, fnames in ((False, list(names_raw)), (True, list(names_norm))):
        Xs, meta = [], {k: [] for k in ("cell_id", "kernel", "archetype", "campaign", "role", "rep", "win_start", "n_series_cell")}
        for c in cells:
            v = vectors.get(c["cell_id"])
            if v is None:
                continue
            vec = np.asarray(v[1] if normalized else v[0], dtype=np.float64)
            Xs.append(vec[None, :])
            k, a = _kernel_label(c)
            meta["cell_id"].append(c["cell_id"]); meta["kernel"].append(k); meta["archetype"].append(a)
            meta["campaign"].append(c["campaign"]); meta["role"].append(c["role"]); meta["rep"].append(int(c["rep"]))
            meta["win_start"].append(0); meta["n_series_cell"].append(int(n_series.get(c["cell_id"], 0)))
        X = np.concatenate(Xs, axis=0) if Xs else np.zeros((0, len(fnames)))
        p = S.features_path(out, name, GRID_ID, normalized)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, X=X, feature_names=np.array(fnames), cell_id=np.array(meta["cell_id"]),
                 kernel=np.array(meta["kernel"]), archetype=np.array(meta["archetype"]),
                 campaign=np.array(meta["campaign"]), role=np.array(meta["role"]),
                 rep=np.array(meta["rep"], dtype=np.int64), win_start=np.array(meta["win_start"], dtype=np.int64),
                 n_series_cell=np.array(meta["n_series_cell"], dtype=np.int64),
                 W=np.array(-1), H=np.array(-1), grid_id=np.array(GRID_ID), normalized=np.array(bool(normalized)),
                 head_drop_json=np.array(hd_json), n_windows_dropped=np.array(int(n_dropped)), wapf_norm=np.array(""))
        paths.append(p)
    return paths[0], paths[1]


def run_comparator_splits(out: Path, name: str, *, null_perm: int = M.B1G1_MIN_PERM, null_splits: str = "loko,loro,within_trace",
                          n_jobs: int = 1, n_estimators: int = M.N_ESTIMATORS, seed_offset: int = 0) -> list[Path]:
    """The split stage of one comparator (SPEC_epoch2 Part 1.6): for ``normalized in (False,
    True)`` and the five (split, label space) pairs ``models splits --all-splits`` uses,
    ``models.run_split_stage(out, "cmp_<name>", "Wall_Hall", split, labelspace, ...)``, which
    applies B1-G1 (the label-shuffle null of ``null_perm`` permutations; P2 Sec. V 5.1 Plan 08;
    SPEC 3.7.1), B1-G3 (the one-feature quarantine, SPEC 3.7.2; the rung's score is the re-run
    through ``models.effective_scores``), B1-G6 (``majority``, SPEC 3.7.3), G-N (the macro recall
    over headline archetypes, SPEC 3.7.5), G-K0's relabelling (at read time, SPEC 3.3.3) and
    G-DIM's ``dim_status`` (SPEC 3.7.7) exactly as for a rung. Within-trace reads ``not
    applicable: one window per cell`` by the split stage's own rule (one row per cell)."""
    out = Path(out)
    ns = set(x for x in str(null_splits).split(",") if x)
    dirs = []
    for normalized in (False, True):
        for split, ls in SPLIT_PLAN:
            dirs.append(M.run_split_stage(out, name, GRID_ID, split, ls, normalized=normalized, n_perm=null_perm,
                                          n_jobs=n_jobs, n_estimators=n_estimators, run_null=split in ns,
                                          seed_offset=seed_offset))
    return dirs


def _split_params(null_perm, null_splits, n_jobs, n_estimators, seed_offset, no_splits) -> dict:
    return {"split_stage": "not run: --no-splits" if no_splits else "models.run_split_stage on both variants and the five (split, labelspace) pairs",
            "null_perm": int(null_perm), "null_splits": str(null_splits), "n_jobs": int(n_jobs), "n_estimators": int(n_estimators),
            "seed_offset": int(seed_offset), "grid_id": GRID_ID, "grid_id_reason": GRID_ID_REASON}


def _inputs_sha(out: Path, extra=()) -> dict:
    out = Path(out)
    return S.inputs_sha256([out / "cells.csv", out / "inputs" / "head_drop.csv", out / "gates" / "preconditions.csv",
                            out / "gates" / "preconditions.json", *extra], out)


# =============================================================================================
# Savoldi 2010 (cmp_savoldi)
# =============================================================================================
def savoldi_cell(ex: dict, head_drop: int, *, rows: str = SAVOLDI_ROWS, ddof: int = SAVOLDI_DDOF, n_pages: int = N) -> dict:
    """Savoldi 2010: U = mu_dmp +/- sigma_dmp, the mean and standard deviation of the per-pair
    changed-page count over the run.

    Citation: C14 cand. 1 (for each consecutive pair the number of differing 4 KiB pages; the
    sample mean mu_dmp and standard deviation sigma_dmp of that count over the run; U = mu_dmp
    +/- sigma_dmp; EXACT on our per-pair K, "only the interval differs"); P2 Sec. 0 Baseline;
    P2E Sec. 7 (the one-number baseline for EUSIPCO); P2 Sec. V through models.run_split_stage
    (run_comparator_splits). In our rows the per-pair count is the extract's ``K`` (every ``seq``
    row is the differ's output for the pair ending at that snapshot, SPEC 2.1, 2.2).
    Parameters the definition leaves open: rows = "all_after_head_drop" ("rung_series": the rows
    the rung series use, the last seq dropped), SPEC_epoch2 Part 4 item 1; ddof = 1 (0: the
    population SD, the toolkit's shape-feature convention), Part 4 item 2.
    Returns the per-cell record: ``n_rows_used, K_mean, K_sd, K_median, U_mean_pct, U_sd_pct,
    U_text, raw`` = (K_mean, K_sd) in pages, ``norm`` = (K_mean / K_median, K_sd / K_median), the
    count-rung rule of P2 Sec. V G-L (i) (Part 4 item 10), and ``status``."""
    if rows not in ("all_after_head_drop", "rung_series"):
        raise ValueError(f"rows must be all_after_head_drop or rung_series, got {rows!r}")
    hd = max(0, int(head_drop))
    K = np.asarray(ex["K"], dtype=np.float64)
    K = K[hd:] if rows == "all_after_head_drop" else K[hd:len(K) - 1]
    n = int(K.shape[0])
    kmed = S.k_median_cell(ex, hd)
    status = "ok"
    if n == 0:
        status = V.not_run("no rows after the head drop")
        k_mean = k_sd = float("nan")
    else:
        k_mean = float(np.mean(K))
        if n < 2:
            k_sd = float("nan")
            status = V.not_run("fewer than two rows")
        else:
            k_sd = float(np.std(K, ddof=int(ddof)))
    u_mean = 100.0 * k_mean / float(n_pages)
    u_sd = 100.0 * k_sd / float(n_pages)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = (k_mean / kmed if kmed and kmed > 0 else float("nan"), k_sd / kmed if kmed and kmed > 0 else float("nan"))
    return {"rows_rule": rows, "ddof": int(ddof), "n_rows_used": n, "K_mean": k_mean, "K_sd": k_sd, "K_median": kmed,
            "U_mean_pct": u_mean, "U_sd_pct": u_sd, "U_text": f"{_pct_text(u_mean, 4)}% +/- {_pct_text(u_sd, 3)}%",
            "raw": (k_mean, k_sd), "norm": norm, "status": status}


SAVOLDI_NAMES_RAW = ("cmp_savoldi.K.mean", "cmp_savoldi.K.sd")
SAVOLDI_NAMES_NORM = ("cmp_savoldi.k_over_med.mean", "cmp_savoldi.k_over_med.sd")
SAVOLDI_NORM_MEANING = ("the first normalized feature is the mean over the median, a number near one that carries only the skew of "
                        "the count distribution; the second is the relative spread; the normalized row tests whether skew and "
                        "relative spread alone carry the label; the raw row is the method as published (level-inclusive by construction)")


def run_savoldi(out: Path, *, rows: str = SAVOLDI_ROWS, ddof: int = SAVOLDI_DDOF, null_perm: int = M.B1G1_MIN_PERM,
                null_splits: str = "loko,loro,within_trace", n_jobs: int = 1, n_estimators: int = M.N_ESTIMATORS,
                seed_offset: int = 0, no_splits: bool = False) -> Path:
    """Savoldi 2010 over every ``ok`` cell with an extract (``savoldi_cell`` per cell): writes
    ``gates/comparators/savoldi.csv`` (+ ``savoldi.params.json``, schema ``plan11.comparators.savoldi.v1``),
    ``savoldi_per_kernel.csv`` (over admissible cells; ``U_text_median`` is the ``U_text`` of the cell
    whose ``K_mean`` is the lower median), ``features/cmp_savoldi/Wall_Hall_{raw,norm}.npz`` and, unless
    ``no_splits``, the split stage (``run_comparator_splits``). Citation: C14 cand. 1; SPEC_epoch2 Part 1.2."""
    out = Path(out)
    cells = _cells_with_extract(out)
    adm, ex_pair, pre_csv, pre_json = _admissibility(out)
    hds, hd_table = _head_drops(out, cells)
    recs, vectors, n_series = [], {}, {}
    n_dropped = 0
    for c in cells:
        ex = S.load_extract_cached(out, c["cell_id"])
        r = savoldi_cell(ex, hds[c["cell_id"]], rows=rows, ddof=ddof)
        rec = {**_identity(c), "admissible": adm.get(c["cell_id"], "true" if pre_csv else "preconditions not run"),
               "head_drop": hds[c["cell_id"]], **{k: r[k] for k in ("rows_rule", "n_rows_used", "K_mean", "K_sd", "K_median",
                                                                    "U_mean_pct", "U_sd_pct", "U_text", "status")}}
        recs.append(rec)
        if r["status"] == "ok":
            vectors[c["cell_id"]] = (r["raw"], r["norm"])
            n_series[c["cell_id"]] = r["n_rows_used"]
        else:
            n_dropped += 1
    d = cmp_dir(out)
    p = S.write_csv(d / "savoldi.csv", SAVOLDI_COLUMNS, recs)
    per_k = []
    for k, rs in _kernel_groups([r for r in recs if r["status"] == "ok" and _admissible_for_summary(r["admissible"])]):
        srt = sorted(rs, key=lambda r: r["K_mean"])
        med_cell = srt[(len(srt) - 1) // 2]
        per_k.append({"kernel": k, "n_cells": len(rs), "K_mean_median": _median_or_nan(r["K_mean"] for r in rs),
                      "K_mean_min": min(r["K_mean"] for r in rs), "K_mean_max": max(r["K_mean"] for r in rs),
                      "K_sd_median": _median_or_nan(r["K_sd"] for r in rs), "U_text_median": med_cell["U_text"]})
    S.write_csv(d / "savoldi_per_kernel.csv", SAVOLDI_KERNEL_COLUMNS, per_k)
    write_feature_files(out, "cmp_savoldi", cells, vectors, names_raw=SAVOLDI_NAMES_RAW, names_norm=SAVOLDI_NAMES_NORM,
                        n_series=n_series, head_drop=hd_table, n_dropped=n_dropped)
    params = {"rows": rows, "ddof": int(ddof), "n_pages": N, "norm_rule": COMPARATOR_NORM_RULE, "norm_row_meaning": SAVOLDI_NORM_MEANING,
              "interval_note": INTERVAL_NOTE, "feature_names_raw": list(SAVOLDI_NAMES_RAW), "feature_names_norm": list(SAVOLDI_NAMES_NORM),
              "n_cells": len(recs), "n_cells_in_features": len(vectors), "n_cells_refused": n_dropped,
              "per_kernel_rule": "admissible cells only (all_hard_pass true; every cell when preconditions.csv is absent); U_text_median = the U_text of the cell whose K_mean is the lower median",
              "preconditions_csv_present": pre_csv, "preconditions_json_present": pre_json, "head_drop": hd_table,
              **_split_params(null_perm, null_splits, n_jobs, n_estimators, seed_offset, no_splits), "inputs_sha256": _inputs_sha(out)}
    S.write_params(p, "plan11.comparators.savoldi.v1", params, CIT_SAVOLDI)
    if not no_splits:
        run_comparator_splits(out, "cmp_savoldi", null_perm=null_perm, null_splits=null_splits, n_jobs=n_jobs,
                              n_estimators=n_estimators, seed_offset=seed_offset)
    return p


# =============================================================================================
# Dhodapkar and Smith 2003 (cmp_dhodapkar)
# =============================================================================================
def _on_grid(value: float, grid) -> bool:
    return any(abs(float(g) - float(value)) <= GRID_TOL for g in grid)


def resolve_grid(grid, default, *, off_grid_rule: str = OFF_GRID_DEFAULT_RULE, what: str = "delta_th_default", flag: str = "--grid") -> tuple[tuple, bool]:
    """(grid, appended): the grid with the default a computed point. A default that is not a grid
    point is refused (``ValueError``, exit 2 on the CLI) under ``off_grid_rule = "refuse"`` or
    appended under ``"append"`` and recorded (SPEC_epoch2_review_al_farabi.md section 2 (b)): the
    feature vector "at the default" is undefined otherwise."""
    grid = tuple(grid)
    if _on_grid(default, grid):
        return grid, False
    if off_grid_rule == "append":
        return tuple(sorted(set(grid) | {default})), True
    raise ValueError(f"{what} {default} is not on the grid {list(grid)} ({flag}; --off-grid-default append to add it)")


def _phase_lengths(boundaries: np.ndarray, n: int, rule: str) -> float:
    """Mean phase length in pairs. ``n_over_b_plus_1``: ``n / (B + 1)``, the mean length of the B + 1
    segments the B boundaries cut the n kept pairs into, the first and last partial segments
    included. ``interior``: a phase runs from one boundary pair up to the pair before the next, so
    the interior segments (bounded by two boundaries) have lengths ``b_{i+1} - b_i`` and the mean
    is ``(b_B - b_1) / (B - 1)``; NaN below two boundaries (Part 4 item 5)."""
    B = int(boundaries.shape[0])
    if rule == "n_over_b_plus_1":
        return float(n) / float(B + 1) if n else float("nan")
    if rule == "interior":
        if B < 2:
            return float("nan")
        return float(boundaries[-1] - boundaries[0]) / float(B - 1)
    raise ValueError(f"phase_length_rule must be n_over_b_plus_1 or interior, got {rule!r}")


def dhodapkar_cell(ex: dict, head_drop: int, *, grid=DHODAPKAR_GRID, default: float = DHODAPKAR_DEFAULT,
                   boundary_rule: str = DHODAPKAR_BOUNDARY_RULE, phase_length_rule: str = DHODAPKAR_PHASE_LENGTH_RULE) -> dict:
    """Dhodapkar and Smith 2003: relative working set distance between consecutive working sets,
    a phase change when it exceeds a threshold, stability and average phase length.

    Citation: C14 cand. 3 (delta_{i,i-1} = (|W_i u W_{i-1}| - |W_i n W_{i-1}|) / |W_i u W_{i-1}|;
    "Their delta is one minus Jaccard, identically"); P2 Sec. 0 Baseline; P2E Sec. 7 (the named
    method on the overlap axis for EUSIPCO); AA 2026-09-17 (a declared sweep, every point kept, 0.04
    marked as the default); C14 author (declare delta_th before the labels or sweep it and show the
    sweep); P2 Sec. V through models.run_split_stage (run_comparator_splits).
    Derivation on our rows: with W_i the changed-page set of pair i, |W_i n W_{i-1}| is the
    extract's ``n_persist`` and |W_i u W_{i-1}| its ``n_union``, so delta = (n_union - n_persist) /
    n_union = 1 - n_persist / n_union = 1 - J (SPEC 2.2 defines J = n_persist / n_union). The
    module computes ``delta = 1 - ex["J"]``, never a second intersection. Rows: ``J[head_drop:-1]``
    (the last seq has no J, SPEC 2.4); a blank J (both sets empty) is dropped and counted in
    ``n_pairs_blank``.
    Parameters the definition leaves open (SPEC_epoch2 Part 4): grid = DHODAPKAR_GRID with default =
    0.04 (item 3; every point computed and kept, ``is_default`` marks the declared one, none selected
    against labels); boundary_rule = "gt" (delta > delta_th; "ge": >=), item 4; phase_length_rule =
    "n_over_b_plus_1" (n / (B + 1); "interior": segments between two boundaries, NaN below two), item 5.
    The default must be a grid point (``resolve_grid``). Returns ``n_pairs_used, n_pairs_blank,
    delta_mean, delta_q05 .. delta_q95``, ``sweep`` (one dict per grid point: delta_th, is_default,
    n_boundaries, stability, mean_phase_length_pairs), ``raw`` = (n_boundaries, stability,
    mean_phase_length_pairs) at the default (the method's three named outputs; two are algebraically
    tied, the method's own redundancy), ``norm`` = (n_boundaries / n_pairs_used, stability,
    mean_phase_length_pairs / n_pairs_used) (the pair count is the only per-cell scale the counts
    carry; the delta itself is level-free), and ``status``."""
    if boundary_rule not in ("gt", "ge"):
        raise ValueError(f"boundary_rule must be gt or ge, got {boundary_rule!r}")
    grid, _ = resolve_grid(grid, default, off_grid_rule="refuse")
    hd = max(0, int(head_drop))
    J = np.asarray(ex["J"], dtype=np.float64)
    J = J[hd:len(J) - 1]
    blank = np.isnan(J)
    delta = 1.0 - J[~blank]
    n = int(delta.shape[0])
    n_blank = int(blank.sum())
    status = "ok" if n > 0 else V.not_run("no pair with a defined J after the head drop")
    if n:
        qs = [float(v) for v in np.quantile(delta, (0.05, 0.25, 0.50, 0.75, 0.95))]
        d_mean = float(np.mean(delta))
    else:
        qs = [float("nan")] * 5
        d_mean = float("nan")
    sweep = []
    default_rec = None
    for th in grid:
        flags = (delta > th) if boundary_rule == "gt" else (delta >= th)
        b_idx = np.flatnonzero(flags)
        B = int(b_idx.shape[0])
        stability = (1.0 - B / n) if n else float("nan")
        mpl = _phase_lengths(b_idx, n, phase_length_rule)
        rec = {"delta_th": float(th), "is_default": _on_grid(th, (default,)), "n_boundaries": B, "stability": stability,
               "mean_phase_length_pairs": mpl}
        sweep.append(rec)
        if rec["is_default"]:
            default_rec = rec
    assert default_rec is not None
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = (default_rec["n_boundaries"] / n if n else float("nan"), default_rec["stability"],
                default_rec["mean_phase_length_pairs"] / n if n else float("nan"))
    return {"n_pairs_used": n, "n_pairs_blank": n_blank, "delta_mean": d_mean, "delta_q05": qs[0], "delta_q25": qs[1],
            "delta_q50": qs[2], "delta_q75": qs[3], "delta_q95": qs[4], "delta_th_default": float(default), "sweep": sweep,
            "n_boundaries": default_rec["n_boundaries"], "stability": default_rec["stability"],
            "mean_phase_length_pairs": default_rec["mean_phase_length_pairs"],
            "raw": (float(default_rec["n_boundaries"]), default_rec["stability"], default_rec["mean_phase_length_pairs"]),
            "norm": norm, "status": status}


DHODAPKAR_NAMES_RAW = ("cmp_dhodapkar.n_boundaries", "cmp_dhodapkar.stability", "cmp_dhodapkar.mean_phase_length")
DHODAPKAR_NAMES_NORM = ("cmp_dhodapkar.boundary_rate", "cmp_dhodapkar.stability", "cmp_dhodapkar.mean_phase_frac")
DHODAPKAR_FEATURE_NOTE = ("the method's three named outputs at the default threshold; stability = 1 - n_boundaries / n_pairs_used and, "
                          "under phase_length_rule = n_over_b_plus_1, mean_phase_length_pairs = n_pairs_used / (n_boundaries + 1): two of "
                          "the three are algebraically tied, the method's own redundancy (B1-G3 may quarantine; SPEC_epoch2 Part 4 item 23)")
DHODAPKAR_NORM_MEANING = ("the delta is level-free (a Jaccard), so no level division exists; the only per-cell scale the counts carry is "
                          "the pair count, and the normalized row removes it: (n_boundaries / n_pairs_used, stability, "
                          "mean_phase_length_pairs / n_pairs_used); the raw row is the method as published")


def run_dhodapkar(out: Path, *, grid=DHODAPKAR_GRID, default: float = DHODAPKAR_DEFAULT, boundary_rule: str = DHODAPKAR_BOUNDARY_RULE,
                  phase_length_rule: str = DHODAPKAR_PHASE_LENGTH_RULE, off_grid_rule: str = OFF_GRID_DEFAULT_RULE,
                  grid_source: str | None = None, null_perm: int = M.B1G1_MIN_PERM, null_splits: str = "loko,loro,within_trace",
                  n_jobs: int = 1, n_estimators: int = M.N_ESTIMATORS, seed_offset: int = 0, no_splits: bool = False) -> Path:
    """Dhodapkar-Smith over every ``ok`` cell with an extract (``dhodapkar_cell`` per cell): writes
    ``gates/comparators/dhodapkar_sweep.csv`` (one row per (cell, delta_th), the sweep whole and never
    filtered), ``dhodapkar.csv`` (the default-threshold row per cell with the delta statistics) and
    ``dhodapkar.params.json`` (schema ``plan11.comparators.dhodapkar.v1``; ``default_source`` and
    ``grid_source`` follow the value that ran, al-Farabi review 2 (a)), ``dhodapkar_per_kernel.csv``
    (medians at the default over admissible cells), ``features/cmp_dhodapkar/Wall_Hall_{raw,norm}.npz``
    and, unless ``no_splits``, the split stage. Citation: C14 cand. 3; AA 2026-09-17; SPEC_epoch2 Part 1.3."""
    out = Path(out)
    grid, appended = resolve_grid(grid, default, off_grid_rule=off_grid_rule, what="delta_th_default", flag="--grid")
    cells = _cells_with_extract(out)
    adm, ex_pair, pre_csv, pre_json = _admissibility(out)
    hds, hd_table = _head_drops(out, cells)
    recs, sweep_rows, vectors, n_series = [], [], {}, {}
    n_dropped = 0
    for c in cells:
        ex = S.load_extract_cached(out, c["cell_id"])
        r = dhodapkar_cell(ex, hds[c["cell_id"]], grid=grid, default=default, boundary_rule=boundary_rule, phase_length_rule=phase_length_rule)
        ident = _identity(c)
        adm_text = adm.get(c["cell_id"], "true" if pre_csv else "preconditions not run")
        exp = ("true" if c["cell_id"] in ex_pair else "false") if pre_json else "preconditions not run"
        rec = {**ident, "admissible": adm_text, "excluded_pair_rung": exp, "head_drop": hds[c["cell_id"]],
               **{k: r[k] for k in ("n_pairs_used", "n_pairs_blank", "delta_mean", "delta_q05", "delta_q25", "delta_q50", "delta_q75",
                                    "delta_q95", "delta_th_default", "n_boundaries", "stability", "mean_phase_length_pairs", "status")}}
        recs.append(rec)
        for s in r["sweep"]:
            sweep_rows.append({"cell_id": ident["cell_id"], "kernel": ident["kernel"], "role": ident["role"], "rep": ident["rep"],
                               "campaign": ident["campaign"], "admissible": adm_text, "excluded_pair_rung": exp,
                               "n_pairs_used": r["n_pairs_used"], "n_pairs_blank": r["n_pairs_blank"], **s})
        if r["status"] == "ok":
            vectors[c["cell_id"]] = (r["raw"], r["norm"])
            n_series[c["cell_id"]] = r["n_pairs_used"]
        else:
            n_dropped += 1
    d = cmp_dir(out)
    S.write_csv(d / "dhodapkar_sweep.csv", DHODAPKAR_SWEEP_COLUMNS, sweep_rows)
    p = S.write_csv(d / "dhodapkar.csv", DHODAPKAR_COLUMNS, recs)
    per_k = []
    for k, rs in _kernel_groups([r for r in recs if r["status"] == "ok" and _admissible_for_summary(r["admissible"])]):
        per_k.append({"kernel": k, "n_cells": len(rs), "n_boundaries_median": _median_or_nan(r["n_boundaries"] for r in rs),
                      "stability_median": _median_or_nan(r["stability"] for r in rs),
                      "mean_phase_length_median": _median_or_nan(r["mean_phase_length_pairs"] for r in rs),
                      "delta_q50_median": _median_or_nan(r["delta_q50"] for r in rs)})
    S.write_csv(d / "dhodapkar_per_kernel.csv", DHODAPKAR_KERNEL_COLUMNS, per_k)
    write_feature_files(out, "cmp_dhodapkar", cells, vectors, names_raw=DHODAPKAR_NAMES_RAW, names_norm=DHODAPKAR_NAMES_NORM,
                        n_series=n_series, head_drop=hd_table, n_dropped=n_dropped)
    is_declared = _on_grid(default, (DHODAPKAR_DEFAULT,))
    params = {"grid": [float(g) for g in grid], "default": float(default), "boundary_rule": boundary_rule, "phase_length_rule": phase_length_rule,
              "default_source": DHODAPKAR_DEFAULT_SOURCE if is_declared else f"CLI --delta-th-default {default} (departs from AA 2026-09-17's {DHODAPKAR_DEFAULT})",
              "default_is_declared": is_declared,
              "grid_source": grid_source or (DHODAPKAR_GRID_SOURCE if tuple(float(g) for g in grid) == tuple(float(g) for g in DHODAPKAR_GRID) else "CLI --grid"),
              "default_appended_to_grid": appended, "off_grid_default_rule": off_grid_rule,
              "delta_definition": "delta = 1 - J = (n_union - n_persist) / n_union (SPEC 2.2); never a second intersection",
              "rows": "J[head_drop:-1], blank J dropped (n_pairs_blank)", "feature_note": DHODAPKAR_FEATURE_NOTE,
              "norm_row_meaning": DHODAPKAR_NORM_MEANING, "norm_rule": "pair count (the delta is level-free)",
              "feature_names_raw": list(DHODAPKAR_NAMES_RAW), "feature_names_norm": list(DHODAPKAR_NAMES_NORM),
              "sweep_file": "gates/comparators/dhodapkar_sweep.csv (every grid point per cell, never filtered)",
              "n_cells": len(recs), "n_cells_in_features": len(vectors), "n_cells_refused": n_dropped,
              "per_kernel_rule": "admissible cells only, medians at the default threshold",
              "preconditions_csv_present": pre_csv, "preconditions_json_present": pre_json, "head_drop": hd_table,
              **_split_params(null_perm, null_splits, n_jobs, n_estimators, seed_offset, no_splits), "inputs_sha256": _inputs_sha(out)}
    S.write_params(p, "plan11.comparators.dhodapkar.v1", params, CIT_DHODAPKAR)
    S.write_params(d / "dhodapkar_sweep.csv", "plan11.comparators.dhodapkar_sweep.v1", params, CIT_DHODAPKAR)
    if not no_splits:
        run_comparator_splits(out, "cmp_dhodapkar", null_perm=null_perm, null_splits=null_splits, n_jobs=n_jobs,
                              n_estimators=n_estimators, seed_offset=seed_offset)
    return p


# =============================================================================================
# Law 2010 (cmp_law): the streaming pass over the trajectory
# =============================================================================================
class _PageList:
    """One finished snapshot's sorted unique page indices (the one page list the Law pass holds;
    tracked in ``_LIVE_PAGE_ARRAYS`` as ``extract.Snapshot`` is in ``_LIVE_SNAPSHOTS``, so the memory
    test can assert that at most one is live at every step; a bare ``numpy.ndarray`` is unhashable
    and cannot be a ``WeakSet`` member, al-Farabi review 5.2)."""

    __slots__ = ("seq", "pages", "__weakref__")

    def __init__(self, seq: int, pages: np.ndarray):
        self.seq = int(seq)
        self.pages = pages
        _LIVE_PAGE_ARRAYS.add(self)


def law_run_length(X: int, x_unit: str = LAW_X_UNIT) -> int:
    """L(X): the membership (or non-membership) run length in consecutive changed sets that makes a
    page dynamic (static) for X: ``X - 1`` under ``"dumps"`` (C14's exact reading: X dumps span X - 1
    pairs) and ``X`` under ``"pairs"`` (the brief's "in every one of the X changed sets"), Part 4 item 7."""
    if x_unit == "dumps":
        return int(X) - 1
    if x_unit == "pairs":
        return int(X)
    raise ValueError(f"x_unit must be dumps or pairs, got {x_unit!r}")


def law_stream(traj_path, *, n_pages: int = N, x_grid=LAW_X_GRID, x_unit: str = LAW_X_UNIT, head_drop: int = 0,
               head_drop_rule: str = LAW_HEAD_DROP_RULE) -> dict:
    """Law et al. 2010: per page the run lengths of consecutive membership and non-membership in the
    changed sets; the count of pages dynamic for X (a membership run of length L(X)) and static for
    X (a non-membership run of length L(X)) at every pair position, for every X of the grid in one
    streaming pass.

    Citation: C14 cand. 2 (a page is dynamic in X consecutive dumps if X or more consecutive hashes
    differ, static if identical; "an index of run lengths answers all X in one pass"; "Dynamic in X"
    = a membership run of length X-1 in our changed sets, "static in X" = a non-membership run of
    length X-1); P2 Sec. 0 Baseline; P2E Sec. 7 (held for IFIP); SPEC_epoch2 Part 1.4 (the pass).
    Input: the trajectory (a second streaming pass in ``extract.py``'s style; ``extract.py`` is not
    edited), read through ``extract.open_text``; the row loop is copied from ``extract._stream``
    reduced to ``seq`` and ``page_index``: the header located by name, a parse failure counted in
    ``n_rows_skipped``, a decreasing ``seq`` refused (``Refusal("seq not monotone at row <n>")``), a
    gap ``seq`` an empty snapshot, duplicate pages within a ``seq`` dropped (``numpy.unique``), a
    page index outside ``0 .. N-1`` refused (the run-length arrays are defined on N pages).
    State: four int32 arrays of length N (``run_m``, ``run_n``, the current membership and
    non-membership run of every page; ``max_m``, ``max_n``, their maxima over the recorded rows) and
    one snapshot's page list; never a page set beyond the current snapshot (stricter than "at most X
    page sets", recorded in ``memory_model``). Per snapshot t (0-based pair index from ``seq_first``,
    gaps included): ``run_m[P] += 1`` and ``run_m[not P] = 0``, ``run_n[P] = 0`` and ``run_n[not P] += 1``;
    ``dyn[X] = count(run_m >= L(X))`` and ``sta[X] = count(run_n >= L(X))`` recorded from ``t_first[X]``
    on. ``dyn_ever[X] = count(max_m >= L(X))``, ``sta_ever[X]`` likewise.
    Parameters the definition leaves open (SPEC_epoch2 Part 4): x_grid = (2, 4, 8, 16) (every point
    computed and kept); x_unit = "dumps" (L = X - 1; "pairs": L = X), item 7; head_drop_rule =
    "runs_from_seq_first" (the run-length state includes the dropped head and recording starts at
    ``t_first = max(head_drop, L - 1)``; "reset_at_head_drop": the four arrays restart at
    ``t = head_drop`` and ``t_first = head_drop + L - 1``, the first window of L pairs after the
    restart), item 8. Returns the per-X series (``dyn``, ``sta`` as int32 arrays), ``t_first``,
    ``L``, ``dyn_ever``, ``sta_ever``, the pass counters and ``memory_model``. Raises ``Refusal``."""
    t0 = time.monotonic()
    Np = int(n_pages)
    xs = [int(x) for x in x_grid]
    if not xs:
        raise ValueError("x_grid is empty")
    L = {X: law_run_length(X, x_unit) for X in xs}
    for X in xs:
        if L[X] < 1:
            raise ValueError(f"X = {X} gives a run length below one under x_unit = {x_unit}")
    if head_drop_rule not in ("runs_from_seq_first", "reset_at_head_drop"):
        raise ValueError(f"head_drop_rule must be runs_from_seq_first or reset_at_head_drop, got {head_drop_rule!r}")
    hd = max(0, int(head_drop))
    if head_drop_rule == "runs_from_seq_first":
        t_first = {X: max(hd, L[X] - 1) for X in xs}
    else:
        t_first = {X: hd + L[X] - 1 for X in xs}
    run_m = np.zeros(Np, dtype=np.int32); run_n = np.zeros(Np, dtype=np.int32)
    max_m = np.zeros(Np, dtype=np.int32); max_n = np.zeros(Np, dtype=np.int32)
    dyn = {X: [] for X in xs}; sta = {X: [] for X in xs}
    stats = {"n_rows_in": 0, "n_rows_skipped": 0, "n_rows_dup_page": 0, "seq_first": None, "seq_last": None,
             "n_seq_present": 0, "gap_seqs": [], "n_seq_gaps": 0, "n_snapshots": 0}
    state = {"t": 0}

    def step(seq: int, pages: np.ndarray) -> None:
        t = state["t"]
        wrapper = _PageList(seq, pages)
        if head_drop_rule == "reset_at_head_drop" and t == hd:
            run_m[:] = 0; run_n[:] = 0; max_m[:] = 0; max_n[:] = 0
        if pages.shape[0]:
            if pages[0] < 0 or pages[-1] >= Np:
                raise Refusal(f"page_index {int(pages[-1] if pages[-1] >= Np else pages[0])} outside 0..{Np - 1} at seq {seq}")
            inc = run_m[pages] + 1                  # a K-length temporary, never an N-length one
            run_m[:] = 0
            run_m[pages] = inc
            np.add(run_n, 1, out=run_n)
            run_n[pages] = 0
        else:
            run_m[:] = 0
            np.add(run_n, 1, out=run_n)
        np.maximum(max_m, run_m, out=max_m)
        np.maximum(max_n, run_n, out=max_n)
        for X in xs:
            if t >= t_first[X]:
                dyn[X].append(int(np.count_nonzero(run_m >= L[X])))
                sta[X].append(int(np.count_nonzero(run_n >= L[X])))
        if LAW_STEP_HOOK is not None:
            LAW_STEP_HOOK(wrapper)
        del wrapper
        state["t"] = t + 1
        stats["n_snapshots"] += 1

    # copied from extract._stream, 2026-09-17, reduced to seq and page_index
    with open_text(str(traj_path)) as fin:
        reader = csv.reader(fin)
        try:
            header = next(reader)
        except StopIteration:
            raise Refusal("empty trajectory file")
        header = [h.strip() for h in header]
        col = {name: i for i, name in enumerate(header)}
        for req in ("seq", "page_index"):
            if req not in col:
                raise Refusal(f"header lacks column {req}")
        i_seq, i_pg = col["seq"], col["page_index"]
        buf_pages: list[int] = []
        cur_seq = None
        gaps: list[int] = stats["gap_seqs"]
        n_present = 0
        for row in reader:
            stats["n_rows_in"] += 1
            try:
                s = int(row[i_seq]); pg = int(row[i_pg])
            except (ValueError, IndexError):
                stats["n_rows_skipped"] += 1
                continue
            if cur_seq is None:
                cur_seq = s
                stats["seq_first"] = s
                n_present = 1
            elif s != cur_seq:
                if s < cur_seq:
                    raise Refusal(f"seq not monotone at row {stats['n_rows_in']}")
                p = np.unique(np.asarray(buf_pages, dtype=np.int64))
                stats["n_rows_dup_page"] += len(buf_pages) - int(p.shape[0])
                buf_pages.clear()
                step(cur_seq, p)
                del p
                for g in range(cur_seq + 1, s):
                    gaps.append(g)
                    step(g, np.empty(0, dtype=np.int64))
                cur_seq = s
                n_present += 1
            buf_pages.append(pg)
        if cur_seq is None:
            raise Refusal("no data rows")
        p = np.unique(np.asarray(buf_pages, dtype=np.int64))
        stats["n_rows_dup_page"] += len(buf_pages) - int(p.shape[0])
        buf_pages.clear()
        step(cur_seq, p)
        del p
        stats["seq_last"] = cur_seq
        stats["n_seq_present"] = n_present
    stats["n_seq_gaps"] = len(gaps)
    dyn_ever = {X: int(np.count_nonzero(max_m >= L[X])) for X in xs}
    sta_ever = {X: int(np.count_nonzero(max_n >= L[X])) for X in xs}
    stats["elapsed_s"] = round(time.monotonic() - t0, 3)
    memory_model = {"n_length_arrays": 4, "dtype": "int32", "nbytes_run_m": int(run_m.nbytes), "nbytes_run_n": int(run_n.nbytes),
                    "nbytes_max_m": int(max_m.nbytes), "nbytes_max_n": int(max_n.nbytes),
                    "nbytes_total_N_length_state": int(run_m.nbytes + run_n.nbytes + max_m.nbytes + max_n.nbytes),
                    "page_lists_live_max": 1,
                    "note": "two run-length arrays and their two maxima, each of length N; one snapshot's page list at a time; "
                            "never a page set beyond the current snapshot (stricter than 'at most X page sets in memory')"}
    return {"status": "ok", "X": xs, "L": L, "x_unit": x_unit, "head_drop": hd, "head_drop_rule": head_drop_rule, "n_pages": Np,
            "t_first": t_first, "dyn": {X: np.asarray(dyn[X], dtype=np.int32) for X in xs},
            "sta": {X: np.asarray(sta[X], dtype=np.int32) for X in xs}, "dyn_ever": dyn_ever, "sta_ever": sta_ever,
            "memory_model": memory_model, **stats}


def _law_series_path(out: Path, cell_id: str) -> Path:
    return cmp_dir(out) / "law_series" / f"{cell_id}.npz"


def _write_law_series(path: Path, res: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    xs = res["X"]
    arrays = {"X": np.array(xs, dtype=np.int64), "t_first": np.array([res["t_first"][X] for X in xs], dtype=np.int64),
              "L": np.array([res["L"][X] for X in xs], dtype=np.int64),
              "dyn_ever": np.array([res["dyn_ever"][X] for X in xs], dtype=np.int64),
              "sta_ever": np.array([res["sta_ever"][X] for X in xs], dtype=np.int64),
              "head_drop": np.array(int(res["head_drop"])), "x_unit": np.array(res["x_unit"]),
              "head_drop_rule": np.array(res["head_drop_rule"]), "n_pages": np.array(int(res["n_pages"])),
              "seq_first": np.array(int(res["seq_first"])), "seq_last": np.array(int(res["seq_last"]))}
    for X in xs:
        arrays[f"dyn_{X}"] = res["dyn"][X]
        arrays[f"sta_{X}"] = res["sta"][X]
    tmp = path.with_suffix(".npz.tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)
    return path


def _load_law_series(path: Path) -> dict:
    d = np.load(path, allow_pickle=False)
    xs = [int(x) for x in d["X"]]
    return {"X": xs, "t_first": {X: int(t) for X, t in zip(xs, d["t_first"])}, "L": {X: int(l) for X, l in zip(xs, d["L"])},
            "dyn_ever": {X: int(v) for X, v in zip(xs, d["dyn_ever"])}, "sta_ever": {X: int(v) for X, v in zip(xs, d["sta_ever"])},
            "dyn": {X: d[f"dyn_{X}"] for X in xs}, "sta": {X: d[f"sta_{X}"] for X in xs},
            "head_drop": int(d["head_drop"]), "x_unit": str(d["x_unit"]), "head_drop_rule": str(d["head_drop_rule"]),
            "n_pages": int(d["n_pages"]), "seq_first": int(d["seq_first"]), "seq_last": int(d["seq_last"])}


def _law_pass_params(x_grid, x_unit, head_drop, head_drop_rule, n_pages) -> dict:
    return {"x_grid": [int(x) for x in x_grid], "x_unit": x_unit, "head_drop": int(head_drop), "head_drop_rule": head_drop_rule, "n_pages": int(n_pages)}


def check_x2_equals_K(res: dict, ex: dict) -> str:
    """The built-in consistency check against ``extract.csv`` (SPEC_epoch2 Part 1.4): at X = 2 under
    ``"dumps"`` (L = 1) the dynamic count is the snapshot's K, so the recorded ``dyn[2]`` must equal
    the extract's ``K`` on the same rows (from ``t_first[2]`` on). ``"true"`` / ``"false"``, or a
    ``not applicable:`` string when X = 2 is not on the grid or the unit is ``pairs``."""
    if res.get("x_unit") != "dumps":
        return V.not_applicable("x_unit = pairs (dyn at X = 2 is n_persist, not K)")
    if 2 not in res["X"]:
        return V.not_applicable("X = 2 not on the grid")
    K = np.asarray(ex["K"], dtype=np.float64)
    t0 = res["t_first"][2]
    d2 = np.asarray(res["dyn"][2], dtype=np.float64)
    K_rows = K[t0:t0 + d2.shape[0]]
    if K_rows.shape[0] != d2.shape[0] or int(ex["_n_rows"]) - t0 != d2.shape[0]:
        return "false"
    return "true" if np.array_equal(K_rows, d2) else "false"


def _law_worker(job: dict) -> dict:
    """One cell of the Law pass (a module-level function so multiprocessing can pickle it; results
    are identical at any job count: the pass is deterministic)."""
    cell_id = job["cell_id"]
    traj = Path(job["traj_path"])
    rec = {"cell_id": cell_id, "traj_file": traj.name, "source_bytes": None, "n_rows_in": 0, "n_rows_skipped": 0, "n_rows_dup_page": 0,
           "seq_first": None, "seq_last": None, "n_seq_gaps": 0, "elapsed_s": None, "status": "ok",
           "pass_params": job["pass_params"], "check_x2_equals_K": None, "memory_model": None}
    try:
        if not traj.is_file():
            rec["status"] = V.not_run(f"trajectory file missing ({traj})")
            return rec
        rec["source_bytes"] = int(traj.stat().st_size)
        res = law_stream(traj, n_pages=job["pass_params"]["n_pages"], x_grid=job["pass_params"]["x_grid"], x_unit=job["pass_params"]["x_unit"],
                         head_drop=job["pass_params"]["head_drop"], head_drop_rule=job["pass_params"]["head_drop_rule"])
    except Refusal as exc:
        rec["status"] = V.refused(str(exc))
        sp = Path(job["series_path"])
        if sp.is_file():
            sp.unlink()
        return rec
    except Exception as exc:  # an internal error is reported per cell, not swallowed
        rec["status"] = f"error: {type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()
        return rec
    for k in ("n_rows_in", "n_rows_skipped", "n_rows_dup_page", "seq_first", "seq_last", "n_seq_gaps", "elapsed_s", "memory_model"):
        rec[k] = res[k]
    ex = S.load_extract(Path(job["out"]), cell_id)
    rec["check_x2_equals_K"] = check_x2_equals_K(res, ex)
    _write_law_series(Path(job["series_path"]), res)
    return rec


def _law_stats(res: dict, X: int, n_pages: int) -> dict:
    d = np.asarray(res["dyn"][X], dtype=np.float64); s = np.asarray(res["sta"][X], dtype=np.float64)
    n = int(d.shape[0])
    if n == 0:
        nan = float("nan")
        return {"n_windows": 0, "dyn_mean": nan, "dyn_sd": nan, "dyn_min": nan, "dyn_max": nan, "dyn_frac_mean": nan,
                "sta_mean": nan, "sta_sd": nan, "sta_min": nan, "sta_max": nan, "sta_frac_mean": nan,
                "dyn_ever": res["dyn_ever"][X], "sta_ever": res["sta_ever"][X],
                "dyn_ever_frac": res["dyn_ever"][X] / n_pages, "sta_ever_frac": res["sta_ever"][X] / n_pages}
    return {"n_windows": n, "dyn_mean": float(d.mean()), "dyn_sd": float(d.std()), "dyn_min": float(d.min()), "dyn_max": float(d.max()),
            "dyn_frac_mean": float(d.mean()) / n_pages, "sta_mean": float(s.mean()), "sta_sd": float(s.std()), "sta_min": float(s.min()),
            "sta_max": float(s.max()), "sta_frac_mean": float(s.mean()) / n_pages, "dyn_ever": res["dyn_ever"][X],
            "sta_ever": res["sta_ever"][X], "dyn_ever_frac": res["dyn_ever"][X] / n_pages, "sta_ever_frac": res["sta_ever"][X] / n_pages}


def law_feature_names(feature_source: str = LAW_FEATURE_SOURCE) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(raw names, normalized names) at the default X (SPEC_epoch2 Part 1.4; Part 4 item 9)."""
    if feature_source == "window":
        return (("cmp_law.dyn.mean", "cmp_law.dyn.sd", "cmp_law.sta.mean", "cmp_law.sta.sd"),
                ("cmp_law.dyn_over_med.mean", "cmp_law.dyn_over_med.sd", "cmp_law.union_over_med.mean", "cmp_law.sta_over_med.sd"))
    if feature_source == "ever":
        return (("cmp_law.dyn.ever", "cmp_law.sta.ever"), ("cmp_law.dyn_over_med.ever", "cmp_law.union_over_med.ever"))
    raise ValueError(f"feature_source must be window or ever, got {feature_source!r}")


def law_features(st: dict, k_median: float, n_pages: int, feature_source: str = LAW_FEATURE_SOURCE) -> tuple[tuple, tuple]:
    """(raw, normalized) feature vectors at the default X from the per-X statistics: ``window``: raw
    (dyn_mean, dyn_sd, sta_mean, sta_sd) in pages, normalized by the count-rung rule with the cell's
    median K (Part 4 item 10): (dyn_mean / K_med, dyn_sd / K_med, (N - sta_mean) / K_med, sta_sd /
    K_med), where N - sta is the union of the window's changed sets, a count; ``ever``: raw
    (dyn_ever, sta_ever), normalized (dyn_ever / K_med, (N - sta_ever) / K_med)."""
    km = float(k_median) if k_median and k_median > 0 else float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        if feature_source == "window":
            raw = (st["dyn_mean"], st["dyn_sd"], st["sta_mean"], st["sta_sd"])
            norm = (st["dyn_mean"] / km, st["dyn_sd"] / km, (n_pages - st["sta_mean"]) / km, st["sta_sd"] / km)
        elif feature_source == "ever":
            raw = (float(st["dyn_ever"]), float(st["sta_ever"]))
            norm = (st["dyn_ever"] / km, (n_pages - st["sta_ever"]) / km)
        else:
            raise ValueError(feature_source)
    return raw, norm


def run_law(out: Path, *, x_grid=LAW_X_GRID, x_default: int = LAW_X_DEFAULT, x_unit: str = LAW_X_UNIT,
            head_drop_rule: str = LAW_HEAD_DROP_RULE, feature_source: str = LAW_FEATURE_SOURCE, off_grid_rule: str = OFF_GRID_DEFAULT_RULE,
            jobs: int = 1, force: bool = False, only: str | None = None, null_perm: int = M.B1G1_MIN_PERM,
            null_splits: str = "loko,loro,within_trace", n_jobs: int = 1, n_estimators: int = M.N_ESTIMATORS, seed_offset: int = 0,
            no_splits: bool = False) -> Path:
    """Law 2010 over every ``ok`` cell with an extract: the streaming pass ``law_stream`` per cell
    (``jobs`` processes, one cell each, as ``extract all --jobs``; resumable per cell: a cell whose
    ``law_cells.json`` entry says ``ok`` with the same pass parameters and an existing series file is
    skipped unless ``force``), then ``gates/comparators/law_sweep.csv`` (one row per (cell, X)),
    ``law.csv`` (the default-X row with ``check_x2_equals_K``), ``law.params.json`` (schema
    ``plan11.comparators.law.v1``; ``x_default_source`` follows the value that ran), ``law_cells.json``
    (the pass counters), ``law_series/<cell_id>.npz`` (``dyn_<X>``, ``sta_<X>`` per recorded pair for
    every grid point), ``law_per_kernel.csv``, ``features/cmp_law/Wall_Hall_{raw,norm}.npz`` and, unless
    ``no_splits``, the split stage. The trajectory path is ``Path(row["path"]) / row["traj_file"]`` of
    ``cells.csv`` (never a hard-coded root). Citation: C14 cand. 2; SPEC_epoch2 Part 1.4."""
    out = Path(out)
    x_grid, appended = resolve_grid([int(x) for x in x_grid], int(x_default), off_grid_rule=off_grid_rule, what="x_default", flag="--x-grid")
    x_grid = tuple(int(x) for x in x_grid)
    law_run_length(x_default, x_unit)                        # validates x_unit
    if head_drop_rule not in ("runs_from_seq_first", "reset_at_head_drop"):
        raise ValueError(f"head_drop_rule must be runs_from_seq_first or reset_at_head_drop, got {head_drop_rule!r}")
    names_raw, names_norm = law_feature_names(feature_source)
    cells = _cells_with_extract(out)
    adm, ex_pair, pre_csv, pre_json = _admissibility(out)
    hds, hd_table = _head_drops(out, cells)
    d = cmp_dir(out)
    cells_json = d / "law_cells.json"
    prev = {}
    if cells_json.is_file():
        try:
            prev = S.read_json(cells_json).get("cells") or {}
        except (OSError, ValueError):
            prev = {}
    rx = re.compile(only) if only else None
    jobs_list, records = [], {}
    for c in cells:
        cid = c["cell_id"]
        pp = _law_pass_params(x_grid, x_unit, hds[cid], head_drop_rule, N)
        sp = _law_series_path(out, cid)
        if rx is not None and not rx.search(cid):
            if cid in prev:
                records[cid] = prev[cid]
            continue
        old = prev.get(cid)
        if not force and old and old.get("status") == "ok" and old.get("pass_params") == pp and sp.is_file():
            records[cid] = {**old, "resumed": True}
            continue
        jobs_list.append({"cell_id": cid, "traj_path": str(Path(c["path"]) / str(c.get("traj_file") or "")), "out": str(out),
                          "series_path": str(sp), "pass_params": pp})
    results = []
    if jobs_list:
        if int(jobs) > 1:
            import multiprocessing as mp
            with mp.Pool(processes=int(jobs)) as pool:
                for res in pool.imap_unordered(_law_worker, jobs_list):
                    results.append(res)
                    print(f"[law] {res['cell_id']}: {res['status']}", file=sys.stderr)
        else:
            for job in jobs_list:
                res = _law_worker(job)
                results.append(res)
                print(f"[law] {res['cell_id']}: {res['status']}", file=sys.stderr)
    for res in results:
        records[res["cell_id"]] = res
    S.write_json(cells_json, "plan11.comparators.law_cells.v1",
                 {"x_grid": list(x_grid), "x_unit": x_unit, "head_drop": "per cell (inputs/head_drop.csv; pass_params of each entry)",
                  "head_drop_rule": head_drop_rule, "n_pages": N, "jobs": int(jobs), "force": bool(force), "only": only, "n_run": len(results),
                  "n_resumed": sum(1 for r in records.values() if r.get("resumed")), "inputs_sha256": _inputs_sha(out)},
                 CIT_LAW, {"cells": {cid: records[cid] for cid in sorted(records)}})
    # ---- the per-cell and per-X statistics from the series files
    sweep_rows, recs, vectors, n_series = [], [], {}, {}
    n_dropped = 0
    L_default = law_run_length(x_default, x_unit)
    for c in cells:
        cid = c["cell_id"]
        ident = _identity(c)
        adm_text = adm.get(cid, "true" if pre_csv else "preconditions not run")
        exp = ("true" if cid in ex_pair else "false") if pre_json else "preconditions not run"
        rec = records.get(cid) or {"status": V.not_run("cell not in this pass (--only)")}
        status = rec.get("status", "ok")
        sp = _law_series_path(out, cid)
        if status == "ok" and not sp.is_file():
            status = V.not_run(f"series file missing ({sp.relative_to(out)})")
        base = {**ident, "admissible": adm_text, "excluded_pair_rung": exp}
        if status != "ok":
            n_dropped += 1
            for X in x_grid:
                sweep_rows.append({**{k: base[k] for k in ("cell_id", "kernel", "role", "rep", "campaign", "admissible", "excluded_pair_rung")},
                                   "X": X, "run_length_L": law_run_length(X, x_unit), "is_default": X == x_default, "status": status})
            recs.append({**base, "head_drop": hds[cid], "X_default": x_default, "run_length_L": L_default,
                         "check_x2_equals_K": rec.get("check_x2_equals_K"), "status": status})
            continue
        res = _load_law_series(sp)
        ex = S.load_extract_cached(out, cid)
        st_default = None
        for X in x_grid:
            st = _law_stats(res, X, N)
            if X == x_default:
                st_default = st
            sweep_rows.append({**{k: base[k] for k in ("cell_id", "kernel", "role", "rep", "campaign", "admissible", "excluded_pair_rung")},
                               "X": X, "run_length_L": res["L"][X], "is_default": X == x_default, **st, "status": "ok"})
        kmed = S.k_median_cell(ex, hds[cid])
        raw, norm = law_features(st_default, kmed, N, feature_source)
        recs.append({**base, "head_drop": hds[cid], "X_default": x_default, "run_length_L": res["L"][x_default],
                     **{k: st_default[k] for k in ("n_windows", "dyn_mean", "dyn_sd", "sta_mean", "sta_sd", "dyn_ever", "sta_ever")},
                     "check_x2_equals_K": rec.get("check_x2_equals_K"), "status": "ok", "_dyn_frac_mean": st_default["dyn_frac_mean"],
                     "_sta_frac_mean": st_default["sta_frac_mean"]})
        if st_default["n_windows"] > 0:
            vectors[cid] = (raw, norm)
            n_series[cid] = st_default["n_windows"]
        else:
            n_dropped += 1
    S.write_csv(d / "law_sweep.csv", LAW_SWEEP_COLUMNS, sweep_rows)
    p = S.write_csv(d / "law.csv", LAW_COLUMNS, recs)
    per_k = []
    for k, rs in _kernel_groups([r for r in recs if r["status"] == "ok" and _admissible_for_summary(r["admissible"])]):
        per_k.append({"kernel": k, "n_cells": len(rs), "dyn_mean_median": _median_or_nan(r["dyn_mean"] for r in rs),
                      "dyn_frac_mean_median": _median_or_nan(r["_dyn_frac_mean"] for r in rs),
                      "sta_frac_mean_median": _median_or_nan(r["_sta_frac_mean"] for r in rs),
                      "dyn_ever_median": _median_or_nan(r["dyn_ever"] for r in rs)})
    S.write_csv(d / "law_per_kernel.csv", LAW_KERNEL_COLUMNS, per_k)
    write_feature_files(out, "cmp_law", cells, vectors, names_raw=names_raw, names_norm=names_norm, n_series=n_series,
                        head_drop=hd_table, n_dropped=n_dropped)
    is_declared = int(x_default) == LAW_X_DEFAULT
    mm = next((r.get("memory_model") for r in records.values() if r.get("memory_model")), None)
    params = {"x_grid": list(x_grid), "x_default": int(x_default), "x_unit": x_unit, "head_drop_rule": head_drop_rule,
              "feature_source": feature_source, "run_length_L_default": L_default,
              "x_default_source": LAW_X_DEFAULT_SOURCE if is_declared else f"CLI --x-default {x_default} (departs from SPEC_epoch2 Part 4 item 6's {LAW_X_DEFAULT})",
              "x_default_is_declared": is_declared,
              "x_grid_source": LAW_X_GRID_SOURCE if tuple(x_grid) == tuple(LAW_X_GRID) else "CLI --x-grid",
              "default_appended_to_grid": appended, "off_grid_default_rule": off_grid_rule, "n_pages": N,
              "memory_model": mm or {"note": "no cell ran in this pass"}, "jobs": int(jobs), "force": bool(force), "only": only,
              "norm_rule": COMPARATOR_NORM_RULE,
              "norm_row_meaning": "the count features divided by the cell's median K (P2 Sec. V G-L (i), the count rung's rule); "
                                  "N - sta is the union of the window's changed sets, a count; the raw row is the method as published",
              "feature_names_raw": list(names_raw), "feature_names_norm": list(names_norm),
              "check_x2_equals_K": "at X = 2 under dumps the dynamic count is K (SPEC_epoch2 Part 1.4); per cell in law.csv",
              "n_cells": len(recs), "n_cells_in_features": len(vectors), "n_cells_refused_or_not_run": n_dropped,
              "per_kernel_rule": "admissible cells only, medians at the default X", "preconditions_csv_present": pre_csv,
              "preconditions_json_present": pre_json, "head_drop": hd_table,
              **_split_params(null_perm, null_splits, n_jobs, n_estimators, seed_offset, no_splits), "inputs_sha256": _inputs_sha(out)}
    S.write_params(p, "plan11.comparators.law.v1", params, CIT_LAW)
    S.write_params(d / "law_sweep.csv", "plan11.comparators.law_sweep.v1", params, CIT_LAW)
    if not no_splits:
        run_comparator_splits(out, "cmp_law", null_perm=null_perm, null_splits=null_splits, n_jobs=n_jobs,
                              n_estimators=n_estimators, seed_offset=seed_offset)
    return p


# =============================================================================================
# The comparison gates on the comparator rows (SPEC_epoch2 Part 1.5)
# =============================================================================================
def _cmp_scores(out: Path, name: str, split: str, ls: str, normalized: bool) -> tuple[dict | None, str]:
    """(the split's scores through ``models.effective_scores``, or None with the ``not run:`` string
    naming the missing file)."""
    d = M.split_dir(out, name, GRID_ID, split, ls, normalized=normalized)
    p = d / "scores.json"
    if not p.is_file():
        return None, V.not_run(f"{p.relative_to(out)} missing (move 14)")
    return M.effective_scores(S.read_json(p)), ""


def _rung_name(name: str, normalized: bool) -> str:
    return name if normalized else f"{name}__raw"


def gl_part1_for(sc: dict | None, missing: str) -> dict:
    """G-L (i) for one comparator (P2 Sec. V 5.2 G-L; SPEC 3.7.4; CR 2.2 item 24): the normalized
    LOKO/archetype score against its B1-G1 null p95, ``pass`` or ``level only``, with the same ``not
    run:`` forms ``gates_comparison.gate_gl`` uses; a ``not applicable:`` split carries its own string."""
    if sc is None:
        return {"verdict": missing or V.not_run("LOKO/archetype scores.json missing")}
    if str(sc.get("b1_g1", "")).startswith("not run"):
        return {"score_norm": sc.get("accuracy"), "null_p95": sc.get("null_p95"), "verdict": V.not_run(str(sc["b1_g1"]).split(": ", 1)[1])}
    if sc.get("accuracy") is None:
        na = str(sc.get("b1_g1") or "")
        return {"verdict": na if na.startswith("not applicable") else V.not_run("LOKO/archetype score missing")}
    v = V.PASS if (sc.get("null_p95") is not None and sc["accuracy"] > sc["null_p95"]) else V.GL_LEVEL_ONLY
    return {"score_norm": sc.get("accuracy"), "null_p95": sc.get("null_p95"), "verdict": v}


def gm_vs_apf(out: Path, name: str, split: str, ls: str, normalized: bool, *, spread, spread_missing: str | None,
              gid_apf: str | None) -> dict:
    """G-M for one comparator row against APF on one split (P2 Sec. V 5.2 G-M; SPEC 3.7.8; CR 2.2 item
    33): ``gates_comparison.gm_compare(scores_cmp, scores_apf, spread)``, the norm variant against APF's
    norm run and the raw variant against APF's raw run, ``spread`` from ``gates/gm.params.json``
    (the measured five-seed spread of move 12; reused, not re-measured, Part 4 item 12). Order of the
    refusals: the comparator's own ``not applicable:`` string (al-Farabi review 5.5), the comparator's
    missing file, ``gates/gm.params.json`` missing (move 12), no selection for apf, APF's split missing."""
    sc, missing = _cmp_scores(out, name, split, ls, normalized)
    row = {"split": split, "rung_a": _rung_name(name, normalized), "rung_b": "apf" if normalized else "apf__raw", "spread": spread}
    if sc is None:
        return {**row, "verdict": missing}
    acc = sc.get("accuracy")
    if isinstance(acc, str) and acc.startswith("not applicable"):
        return {**row, "score_a": None, "verdict": acc}
    if acc is None:
        na = str(sc.get("b1_g1") or "")
        return {**row, "verdict": na if na.startswith("not ") else V.not_run("the comparator's score is missing")}
    row["score_a"] = acc
    if spread_missing:
        return {**row, "verdict": spread_missing}
    if gid_apf is None:
        return {**row, "verdict": V.not_run("no selection for apf")}
    pa = M.split_dir(out, "apf", gid_apf, split, ls, normalized=normalized) / "scores.json"
    if not pa.is_file():
        return {**row, "verdict": V.not_run(f"{pa.relative_to(out)} missing")}
    sa = M.effective_scores(S.read_json(pa))
    row["score_b"] = sa.get("accuracy")
    return {**row, **GX.gm_compare(sc, sa, spread)}


def run_gates(out: Path, *, null_perm: int = M.B1G1_MIN_PERM, n_jobs: int = 1, n_estimators: int = M.N_ESTIMATORS, seed_offset: int = 0) -> Path:
    """G-L (i), G-DIM, G-M vs APF, G-X and the ``verdicts.csv`` summary for every comparator whose
    split directories exist; a missing comparator gets ``not run:`` rows (SPEC_epoch2 Part 1.5, 1.6).
    Citation: P2 Sec. V 5.2 (G-L, G-DIM, G-M, G-X as SPEC 3.7.4, 3.7.7, 3.7.8, 3.7.6 define them);
    the gates that do not apply to a comparator row print the fixed strings of Part 1.5 (module
    constants NA_GC, NA_GF1, NA_GP, NA_GDEC, NA_TEMPORAL) and are not computed here.
    Files: ``gates/comparators/gl.csv`` (``gates_comparison.GL_COLUMNS``; one part (i) row per
    comparator, no part (ii), ``gl.params.json`` says why), ``gdim.csv`` (the columns of
    ``gates/gdim.csv``; rows ``cmp_<name>`` and ``cmp_<name>__raw``, ``d_matched = d``, no matched row:
    a comparator is not the combined rung), ``gm.csv`` (the columns of ``gates/gm.csv``), G-X through
    ``gates_comparison.gate_gx(out, "cmp_<name>", ..., grid_id="Wall_Hall")`` into ``gates/gx.csv``
    (per-rung replacement) and ``gates/gx_runs/cmp_<name>/``, and ``verdicts.csv``."""
    out = Path(out)
    d = cmp_dir(out)
    gid_apf, _ = S.selected_grid_id(out, "apf", None)
    gm_params_p = out / "gates" / "gm.params.json"
    spread, spread_missing = None, None
    if gm_params_p.is_file():
        try:
            spread = (S.read_json(gm_params_p).get("params") or {}).get("spread")
        except (OSError, ValueError):
            spread = None
        if spread is None:
            spread_missing = V.not_run("gates/gm.params.json has no params.spread (move 12)")
    else:
        spread_missing = V.not_run("gates/gm.params.json missing (move 12)")
    gl_rows, gdim_rows, gm_rows, gx_text, verdict_rows = [], [], [], {}, []
    for name in schema.COMPARATOR_NAMES:
        # G-L (i): the norm variant's LOKO/archetype score against its null p95
        sc_norm, miss_norm = _cmp_scores(out, name, "loko", "archetype", True)
        gl_rows.append({"rung": name, "part": "i", "grid_id": GRID_ID, **gl_part1_for(sc_norm, miss_norm)})
        # G-DIM: dim_status per variant, no matched row
        for normalized in (True, False):
            sc, miss = _cmp_scores(out, name, "loko", "archetype", normalized)
            if sc is None:
                gdim_rows.append({"rung": _rung_name(name, normalized), "grid_id": GRID_ID, "status": miss, "d_matched": "", "method": "", "matched_to": ""})
            else:
                gdim_rows.append({"rung": _rung_name(name, normalized), "grid_id": GRID_ID, "d": sc.get("feature_count"),
                                  "d_matched": sc.get("feature_count"), "method": "", "status": sc.get("dim_status") or V.GDIM_FULL,
                                  "loko_score": sc.get("accuracy"), "matched_to": ""})
        # G-M vs APF: like against like on the three splits
        for normalized in (True, False):
            for split, ls in GM_SPLITS:
                gm_rows.append(gm_vs_apf(out, name, split, ls, normalized, spread=spread, spread_missing=spread_missing, gid_apf=gid_apf))
        # G-X on the norm features (as for the rungs)
        fp = S.features_path(out, name, GRID_ID, True)
        if fp.is_file():
            GX.gate_gx(out, name, n_perm=null_perm, n_jobs=n_jobs, n_estimators=n_estimators, seed_offset=seed_offset, grid_id=GRID_ID)
            gx_row = next((r for r in S.read_csv(out / "gates" / "gx.csv") if r.get("rung") == name), None)
            gx_text[name] = (gx_row["leak_verdict"] + (f"; {gx_row['confound_verdict']}" if gx_row.get("confound_verdict") else "")) if gx_row else V.not_run("gates/gx.csv has no row for " + name)
        else:
            gx_text[name] = V.not_run(f"{fp.relative_to(out)} missing (move 14)")
    gl_p = S.write_csv(d / "gl.csv", GX.GL_COLUMNS, gl_rows)
    S.write_params(gl_p, "plan11.comparators.gl.v1", {"part_ii": GL_PART_II, "score_source": GX.SCORE_SOURCE, "grid_id": GRID_ID,
                                                       "inputs_sha256": _inputs_sha(out)}, CIT_GATES)
    gdim_p = S.write_csv(d / "gdim.csv", GDIM_COLUMNS, gdim_rows)
    S.write_params(gdim_p, "plan11.comparators.gdim.v1", {"matched_row": V.not_applicable("a comparator is not the combined rung (no feature-count-matched comparison)"),
                                                           "d_matched_rule": "d_matched = d, method empty", "score_source": GX.SCORE_SOURCE,
                                                           "inputs_sha256": _inputs_sha(out)}, CIT_GATES)
    gm_p = S.write_csv(d / "gm.csv", GM_COLUMNS, gm_rows)
    S.write_params(gm_p, "plan11.comparators.gm.v1", {"spread": spread, "spread_source": "gates/gm.params.json params.spread (move 12; reused, not re-measured; Part 4 item 12)",
                                                       "spread_status": spread_missing or "ok", "grid_id_apf": gid_apf, "like_against_like": "norm vs apf, raw vs apf__raw",
                                                       "sign_rule": [list(x) for x in GX.GM_SIGN_RULE], "score_source": GX.SCORE_SOURCE,
                                                       "inputs_sha256": _inputs_sha(out, [gm_params_p, out / "gates" / "selection.json"])}, CIT_GATES)
    # ---- verdicts.csv: one row per (comparator, variant, split, labelspace)
    gl_by = {r["rung"]: r.get("verdict", "") for r in gl_rows}
    gdim_by = {r["rung"]: r.get("status", "") for r in gdim_rows}
    gm_by = {(r["rung_a"], r["split"]): r.get("verdict", "") for r in gm_rows}
    for name in schema.COMPARATOR_NAMES:
        for normalized in (False, True):
            rn = _rung_name(name, normalized)
            for split, ls in SPLIT_PLAN:
                sc, miss = _cmp_scores(out, name, split, ls, normalized)
                row = {"rung": name, "variant": "norm" if normalized else "raw", "split": split, "labelspace": ls,
                       "gl": (gl_by.get(name, "") if normalized else GL_RAW_TEXT), "gdim": gdim_by.get(rn, ""),
                       "gm_vs_apf": gm_by.get((rn, split), V.not_applicable(f"G-M is read in the split's primary label space ({PRIMARY_LABELSPACE[split]})")),
                       "gx": gx_text.get(name, "")}
                if sc is None:
                    row.update({k: miss for k in ("feature_count", "accuracy", "macro_recall", "null_p95", "rank", "majority", "b1_g1")})
                    row["status"] = miss
                else:
                    fc = sc.get("feature_count_used") if sc.get("feature_count_used") is not None else sc.get("feature_count")
                    row.update({"feature_count": fc, "accuracy": sc.get("accuracy"), "macro_recall": sc.get("macro_recall"),
                                "null_p95": sc.get("null_p95"), "rank": sc.get("b1_g1_rank"), "majority": sc.get("majority"),
                                "b1_g1": sc.get("b1_g1"), "status": sc.get("status", "ok")})
                verdict_rows.append(row)
    vp = S.write_csv(d / "verdicts.csv", VERDICT_COLUMNS, verdict_rows)
    S.write_params(vp, "plan11.comparators.verdicts.v1", {"score_source": GX.SCORE_SOURCE, "grid_id": GRID_ID, "not_applied": {
        "G-C": NA_GC, "G-F (i)": NA_GF1, "G-P": NA_GP, "G-DEC": NA_GDEC, "temporal gates, G-ORD, the grid": NA_TEMPORAL,
        "G-J, G-V, clustering, G-F (ii)": V.not_run("not computed for comparators in this epoch (SPEC_epoch2 Part 4 item 13)")},
        "inputs_sha256": _inputs_sha(out)}, CIT_GATES)
    return vp


# =============================================================================================
# CLI
# =============================================================================================
def _add_split_flags(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--null-perm", type=int, default=M.B1G1_MIN_PERM)
    sp.add_argument("--null-splits", default="loko,loro,within_trace")
    sp.add_argument("--n-jobs", type=int, default=1)
    sp.add_argument("--n-estimators", type=int, default=M.N_ESTIMATORS)
    sp.add_argument("--seed-offset", type=int, default=0)


def _add_savoldi_flags(sp):
    sp.add_argument("--rows", default=SAVOLDI_ROWS, choices=("all_after_head_drop", "rung_series"))
    sp.add_argument("--ddof", type=int, default=SAVOLDI_DDOF)


def _add_dhodapkar_flags(sp):
    sp.add_argument("--grid", default=",".join(str(g) for g in DHODAPKAR_GRID), help="comma-separated delta_th points")
    sp.add_argument("--delta-th-default", type=float, default=DHODAPKAR_DEFAULT)
    sp.add_argument("--boundary-rule", default=DHODAPKAR_BOUNDARY_RULE, choices=("gt", "ge"))
    sp.add_argument("--phase-length-rule", default=DHODAPKAR_PHASE_LENGTH_RULE, choices=("n_over_b_plus_1", "interior"))


def _add_law_flags(sp):
    sp.add_argument("--x-grid", default=",".join(str(x) for x in LAW_X_GRID), help="comma-separated X points")
    sp.add_argument("--x-default", type=int, default=LAW_X_DEFAULT)
    sp.add_argument("--x-unit", default=LAW_X_UNIT, choices=("dumps", "pairs"))
    sp.add_argument("--head-drop-rule", default=LAW_HEAD_DROP_RULE, choices=("runs_from_seq_first", "reset_at_head_drop"))
    sp.add_argument("--feature-source", default=LAW_FEATURE_SOURCE, choices=("window", "ever"))
    sp.add_argument("--jobs", type=int, default=1)
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--only", default=None, help="regex on cell_id (the Law pass only)")


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="comparators.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = []
    for cmd in ("savoldi", "dhodapkar", "law", "gates", "all"):
        sp = sub.add_parser(cmd)
        sp.add_argument("--out", required=True)
        if cmd in ("savoldi", "all"):
            _add_savoldi_flags(sp)
        if cmd in ("dhodapkar", "all"):
            _add_dhodapkar_flags(sp)
        if cmd in ("law", "all"):
            _add_law_flags(sp)
        if cmd in ("dhodapkar", "law", "all"):
            sp.add_argument("--off-grid-default", default=OFF_GRID_DEFAULT_RULE, choices=("refuse", "append"),
                            help="a declared default that is not a grid point: refuse (exit 2) or append it to the grid (recorded)")
        _add_split_flags(sp)
        if cmd != "gates":
            sp.add_argument("--no-splits", action="store_true", help="the statistics and the feature files only")
        common.append(sp)
    return ap


def _parse_grid(text: str, cast):
    return tuple(cast(x.strip()) for x in str(text).split(",") if x.strip())


def main(argv: list[str] | None = None) -> int:
    ap = _build_parser()
    a = ap.parse_args(argv)
    out = Path(a.out)
    if not (out / "cells.csv").is_file():
        print(f"missing input: {out / 'cells.csv'}", file=sys.stderr)
        return 2
    split_kw = dict(null_perm=a.null_perm, null_splits=a.null_splits, n_jobs=a.n_jobs, n_estimators=a.n_estimators, seed_offset=a.seed_offset)
    try:
        if a.cmd in ("savoldi", "all"):
            print(run_savoldi(out, rows=a.rows, ddof=a.ddof, no_splits=a.no_splits, **split_kw))
        if a.cmd in ("dhodapkar", "all"):
            print(run_dhodapkar(out, grid=_parse_grid(a.grid, float), default=a.delta_th_default, boundary_rule=a.boundary_rule,
                                phase_length_rule=a.phase_length_rule, off_grid_rule=a.off_grid_default,
                                no_splits=a.no_splits, **split_kw))
        if a.cmd in ("law", "all"):
            print(run_law(out, x_grid=_parse_grid(a.x_grid, int), x_default=a.x_default, x_unit=a.x_unit, head_drop_rule=a.head_drop_rule,
                          feature_source=a.feature_source, off_grid_rule=a.off_grid_default, jobs=a.jobs, force=a.force, only=a.only,
                          no_splits=a.no_splits, **split_kw))
        if a.cmd == "gates" or (a.cmd == "all" and not a.no_splits):
            # `all --no-splits` is the inspection path (statistics and feature files only): the gates read
            # the split stage and G-X would fit a forest, so they are skipped with it
            print(run_gates(out, null_perm=a.null_perm, n_jobs=a.n_jobs, n_estimators=a.n_estimators, seed_offset=a.seed_offset))
    except FileNotFoundError as e:
        print(f"missing input: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        if "is not on the grid" in str(e):
            print(f"missing input: {e}", file=sys.stderr)
            return 2
        traceback.print_exc()
        return 1
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
