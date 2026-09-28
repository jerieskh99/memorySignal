#!/usr/bin/env python3
"""schema.py -- every shared constant and column list of the plan11 toolkit (SPEC section 1,
builder 1). Builders 2 and 3 import this module and never edit it.

Sources cited: `apf_paper/P2_STRUCTURE.md` (P2), `apf_paper/council/12_P2_COUNCIL_REPORT.md`
section 2 (CR), `apf_paper/council/10_al_kindi_revised.md` (K2), `apf_paper/P2_AUTHOR_ANSWERS.md`
(AA), and `plan11_encoding_ladder/SPEC.md` (SPEC).

No server path appears here. No sandbox workload is named here.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# The instrument (P2 Table 1; SPEC 1)
# ---------------------------------------------------------------------------
N_PAGES = 262144                 # 1024 MiB / 4 KiB (P2 Sec. IV rung 0; Table 1)
PAGE_SIZE = 4096                 # bytes per page
BITS_PER_PAGE = 32768            # PAGE_SIZE * 8 (wAPF denominator, P2 Sec. IV rung 0')
DURATION_S = 600                 # guest-running seconds per cell (AA A6, S5; Table 1)
DT_BRACKET_S = (0.500, 0.644)    # configured interval and derived guest spacing (CR 2.3 item 37)
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)   # the fixed quantiles (SPEC 2.2; CR 2.1 item 17)
QUANTILE_TAGS = ("q05", "q25", "q50", "q75", "q95")
CHANNELS = ("ham", "l0", "l1")   # extract channel prefixes for hamming, l0, l1 (SPEC 2.2)
HEADER_NCOLS_EXPECTED = 66       # seq, page_index + the 64 differ metrics (SPEC 2.1)
TRAJ_GLOB = "*substrate_trajectory.csv*"    # SPEC 2.1; plan10_analysis/corpus_manifest.py
SIDECAR_SCHEMA = "plan11.extract.v1"
CELLS_SCHEMA = "plan11.cells.v1"
SYNTH_TRUTH_SCHEMA = "plan11.synth_truth.v1"

# The 64 differ metric names in row order (live_delta_calc_modular/src/metrics/mod.rs
# `csv_header`, read 2026-09-16). The trajectory header is "seq,page_index," + these.
DIFFER_METRIC_NAMES = (
    "hamming", "cosine", "l0", "l1", "l2", "linf", "mean_abs", "gradient_mag", "tv", "chi2",
    "hellinger", "kl", "js", "wasserstein", "bhattacharyya", "hist_inter_dist", "ent_delta",
    "csize_delta", "ncd", "struct_ent_change", "lz_change", "pearson", "ssim_struct",
    "mean_shift", "ssim_lum", "median_shift", "var_ratio", "std_delta", "ssim_contrast",
    "range_delta", "iqr_delta", "polarity", "sign_delta_ent", "zero_mass_delta", "spearman",
    "kendall", "cross_corr_lag", "phase_corr", "byte_rotation", "ent_q", "struct_ent_q",
    "distinct_bytes", "zero_frac", "fill_frac", "printable_frac", "mean_q", "var_q", "skew_q",
    "kurt_q", "chi2_uniform", "bigram_ent", "autocorr_peak", "changed_runs", "change_span",
    "change_centroid", "longest_changed_run", "change_density", "edge_energy", "glcm_contrast",
    "glcm_homogeneity", "glcm_energy", "glcm_correlation", "high_freq_frac", "max_run_len",
)
TRAJ_HEADER = ("seq", "page_index") + DIFFER_METRIC_NAMES
assert len(TRAJ_HEADER) == HEADER_NCOLS_EXPECTED
TRAJ_HEADER_LINE = ",".join(TRAJ_HEADER)

# ---------------------------------------------------------------------------
# The extract columns (SPEC 2.2), 58 in this fixed order
# ---------------------------------------------------------------------------
def _channel_block(suffix: str) -> tuple[str, ...]:
    cols: list[str] = []
    for ch in CHANNELS:
        cols.append(f"{ch}_sum_{suffix}")
        for q in QUANTILE_TAGS:
            cols.append(f"{ch}_{q}_{suffix}")
    return tuple(cols)


RATIO_PREFIXES = ("r_l0", "r_l1l0", "r_haml0")
RATIO_COLUMNS = tuple(f"{p}_{q}_per" for p in RATIO_PREFIXES for q in QUANTILE_TAGS)

EXTRACT_COLUMNS: tuple[str, ...] = (
    ("seq", "K", "n_persist", "n_union", "J", "J_null_inter", "J_null")
    + _channel_block("all")
    + _channel_block("per")
    + RATIO_COLUMNS
)
assert len(EXTRACT_COLUMNS) == 58, len(EXTRACT_COLUMNS)

# Columns written as integers; every other numeric column is a float written with
# format(x, ".10g"); blanks are the empty string (SPEC 2.2).
EXTRACT_INT_COLUMNS = frozenset(
    ("seq", "K", "n_persist", "n_union")
    + tuple(f"{ch}_sum_{s}" for ch in CHANNELS for s in ("all", "per"))
)

# ---------------------------------------------------------------------------
# The cells (P2 Table 3; AA A2, A4, A7)
# ---------------------------------------------------------------------------
# Ordered (kernel, predicted archetype). P2 Table 3; AA A2.
KERNELS: tuple[tuple[str, str], ...] = (
    ("gemm", "WORKING-SET"), ("floyd", "WORKING-SET"), ("gibbs", "WORKING-SET"),
    ("nbody", "WORKING-SET"), ("spmm", "WORKING-SET"), ("stencil_jacobi", "WORKING-SET"),
    ("fft", "SCATTER"), ("histogram", "SCATTER"), ("fem_assembly", "SCATTER"),
    ("lexer", "SEQUENTIAL-GROW"), ("rmat_gen", "SEQUENTIAL-GROW"),
    ("bnb_tsp", "FRONTIER-CHURN"),
)
KERNEL_NAMES: tuple[str, ...] = tuple(k for k, _ in KERNELS)
ARCHETYPE_OF: dict[str, str] = dict(KERNELS)
ARCHETYPES = ("IDLE", "WORKING-SET", "SCATTER", "SEQUENTIAL-GROW", "FRONTIER-CHURN")
# P2 Sec. 2 and Sec. VI; AA A7: floyd, histogram, nbody at 2,048 pages; fft and gemm at 4,096.
LEVEL_MATCHED_SETS = (("floyd", "histogram", "nbody"), ("fft", "gemm"))

# The paper's rep index (P2 Sec. VI; AA A4): rep 0 is seed 42; reps 1..7 ascend by seed.
REP0_SEED = 42

# Idle-role detection: substrings of the test label, case-insensitive (SPEC 2.6; section 8 item 5).
IDLE_MARKERS_DEFAULT: tuple[str, ...] = ("sleep", "idle")

# ---------------------------------------------------------------------------
# The declared temporal grid (P2 Sec. V 5.1 Plan 03; K2 Sec. 2 rung 0 (b); SPEC 3.1.3)
# ---------------------------------------------------------------------------
GRID_WINDOWS: tuple = (8, 16, 32, 64, "whole")
GRID_HOP_RATIOS: tuple[float, ...] = (0.25, 0.50, 1.00)
WHOLE = "whole"
GRID_ID_WHOLE = "Wall_Hall"
INTEGER_WINDOWS: tuple[int, ...] = tuple(w for w in GRID_WINDOWS if w != WHOLE)


def hop_of(W: int, ratio: float) -> int:
    """H = max(1, round(W * ratio)), the Plan 03 hop rule (`plan03_sweep.py` defaults; SPEC 3.1.3)."""
    return max(1, int(round(int(W) * float(ratio))))


def grid_id(W, H) -> str:
    """The on-disk id of a grid point (SPEC 3.1.3): `W{W}_H{H}` for the twelve integer points,
    `Wall_Hall` for the whole-cell point. The whole-cell point may be passed as
    (`"whole"`, anything), as (None, None), or as (n_series, n_series): the declared grid has no
    integer W outside (8, 16, 32, 64), so an integer W outside that set with H == W is the
    whole-cell point by construction. Any other (W, H) is not on the declared grid and raises
    ValueError, so a mis-typed point can never be written under a grid id."""
    if W == WHOLE or H == WHOLE or W is None or H is None:
        return GRID_ID_WHOLE
    W = int(W)
    H = int(H)
    if W in INTEGER_WINDOWS:
        if H not in {hop_of(W, r) for r in GRID_HOP_RATIOS}:
            raise ValueError(f"H={H} is not a declared hop for W={W} (ratios {GRID_HOP_RATIOS})")
        return f"W{W}_H{H}"
    if H == W and W >= 1:
        return GRID_ID_WHOLE
    raise ValueError(f"(W={W}, H={H}) is not on the declared grid")


def grid_points(n_series: int | None = None) -> list[tuple]:
    """The 13 declared (W, H) pairs in fixed order (SPEC 3.1.3): W in (8, 16, 32, 64) each with
    H = max(1, round(W * r)) for r in (0.25, 0.50, 1.00), then the whole-cell point
    (n_series, n_series), or ("whole", "whole") when `n_series` is None. `grid_id` maps each
    pair to its id."""
    pts: list[tuple] = []
    for W in GRID_WINDOWS:
        if W == WHOLE:
            if n_series is None:
                pts.append((WHOLE, WHOLE))
            else:
                n = int(n_series)
                pts.append((n, n))
        else:
            for r in GRID_HOP_RATIOS:
                pts.append((int(W), hop_of(W, r)))
    assert len(pts) == 13
    return pts


def hop_ratio_of(W: int, H: int) -> float:
    """H / W as a float (Table 5's `hop_ratio` column)."""
    return float(H) / float(W) if W else float("nan")


# ---------------------------------------------------------------------------
# Cell identity (SPEC 2.6, 2.7; P2 Table 1 labels; AA S1)
# ---------------------------------------------------------------------------
_RE_SEED = re.compile(r"seed_(\d+)")            # plan10_analysis/results_view.py line 44
_RE_REP_DIR = re.compile(r"^rep(\d{3})")
_RE_KERNEL_PREFIX = re.compile(r"^kernel_")
_RE_VERSION_SUFFIX = re.compile(r"_v\d+$")


def campaign_of(label: str) -> str:
    """Campaign of a retention label (SPEC 2.6; P2 Table 1; AA S1): labels starting with
    `dwarfs1` (the launch and its resumes) -> "dwarfs1"; the two deep-dive launch labels ->
    "01c" and "01c1"; anything else is returned unchanged."""
    label = str(label or "")
    if label.startswith("dwarfs1"):
        return "dwarfs1"
    if label == "sandbox_deepdive_01c":
        return "01c"
    if label == "sandbox_deepdive_01c1":
        return "01c1"
    return label


def kernel_of_test_label(test_label: str) -> str:
    """`kernel_gemm_v2 -> gemm`, `kernel_stencil_jacobi_v2 -> stencil_jacobi` (SPEC 2.6): a
    leading `kernel_` and a trailing `_v<digits>` removed."""
    k = _RE_KERNEL_PREFIX.sub("", str(test_label))
    k = _RE_VERSION_SUFFIX.sub("", k)
    return k


def parse_cell_path(path, idle_markers: tuple[str, ...] = IDLE_MARKERS_DEFAULT) -> dict:
    """Cell identity from the last four path components of a retention cell directory
    `<family>/<test_label>/<param-sig>/rep<NNN>__<label>/` (SPEC 2.6; verified against
    `run_files_controlled.py` `retention_workload_path` and `plan10_analysis/results_view.py`
    line 109). Returns family, test_label, kernel, param_sig, seed (int or None), rep_dir (int or
    None), label, campaign, role ("idle" | "kernel" | "unknown") and archetype_predicted
    (P2 Table 3 for a kernel; "control" for idle; "unknown" otherwise). The paper's rep index is
    NOT assigned here (see `extract.build_index`)."""
    parts = [p for p in Path(str(path)).parts if p not in ("", "/")]
    if len(parts) < 4:
        raise ValueError(f"cell path needs at least four components: {path!r}")
    family, test_label, param_sig, rep_comp = parts[-4], parts[-3], parts[-2], parts[-1]
    kernel = kernel_of_test_label(test_label)
    m_seed = _RE_SEED.search(param_sig)
    seed = int(m_seed.group(1)) if m_seed else None
    m_rep = _RE_REP_DIR.match(rep_comp)
    rep_dir = int(m_rep.group(1)) if m_rep else None
    label = rep_comp.split("__", 1)[1] if "__" in rep_comp else ""
    tl_lower = test_label.lower()
    if any(str(mk).lower() in tl_lower for mk in idle_markers):
        role = "idle"
        archetype = "control"
    elif kernel in ARCHETYPE_OF:
        role = "kernel"
        archetype = ARCHETYPE_OF[kernel]
    else:
        role = "unknown"
        archetype = "unknown"
    return {
        "family": family,
        "test_label": test_label,
        "kernel": kernel,
        "param_sig": param_sig,
        "seed": seed,
        "rep_dir": rep_dir,
        "label": label,
        "campaign": campaign_of(label),
        "role": role,
        "archetype_predicted": archetype,
    }


def cell_id_of(kernel: str, role: str, rep: int, campaign: str) -> str:
    """`{kernel}__rep{rep:02d}__{campaign}` for kernels, `idle__rep{rep:02d}__{campaign}` for
    idle cells (SPEC 2.6). Filesystem-safe by construction."""
    head = "idle" if role == "idle" else str(kernel)
    return f"{head}__rep{int(rep):02d}__{campaign}"


# cells.csv columns (SPEC 2.7)
CELLS_COLUMNS = (
    "cell_id", "kernel", "role", "archetype_predicted", "seed", "rep", "rep_dir", "label",
    "campaign", "path", "traj_file", "status",
)

# Status strings of cells.csv (SPEC 2.7)
STATUS_OK = "ok"
STATUS_TRAJ_COUNT = "refused: trajectory file count != 1"
STATUS_DUP_SEED = "refused: duplicate seed"
STATUS_UNKNOWN_KERNEL = "refused: unknown kernel"


def format_value(name: str, value) -> str:
    """Serialize one extract cell (SPEC 2.2): integers as integers, floats with
    format(x, ".10g"), blanks (None) as the empty string."""
    if value is None:
        return ""
    if name in EXTRACT_INT_COLUMNS:
        return str(int(value))
    return format(float(value), ".10g")


# ---------------------------------------------------------------------------
# Exact-input comparators (build epoch 2, SPEC_epoch2.md Part 1; C14 candidates 1, 3, 2)
# ---------------------------------------------------------------------------
COMPARATORS: tuple[tuple[str, str], ...] = (
    ("cmp_savoldi", "Savoldi 2010"),
    ("cmp_dhodapkar", "Dhodapkar-Smith 2003"),
    ("cmp_law", "Law 2010"),
)
COMPARATOR_NAMES: tuple[str, ...] = tuple(k for k, _ in COMPARATORS)
COMPARATOR_DISPLAY: dict[str, str] = dict(COMPARATORS)
COMPARATOR_GRID_ID = "Wall_Hall"      # one row per cell: the whole-cell point, by definition
# The declared default of each comparator's free parameter, as (parameter name, value): the
# Dhodapkar-Smith threshold is AA 2026-09-17 ("0.04 marked as the default"); Law's X is
# SPEC_epoch2 Part 4 item 6 (the definition names no X; the author has declared none). tables.py
# reads these to print "declared default" only when the value that ran equals the declared one
# (SPEC_epoch2_review_al_farabi.md section 2 (a)); comparators.py asserts its own constants equal them.
COMPARATOR_DECLARED_DEFAULTS: dict[str, tuple[str, float | int]] = {
    "cmp_dhodapkar": ("delta_th", 0.04),
    "cmp_law": ("X", 4),
}
