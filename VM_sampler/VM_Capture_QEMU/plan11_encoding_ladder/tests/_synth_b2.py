#!/usr/bin/env python3
"""_synth_b2.py -- builder 2's minimal in-test generator (SPEC section 5, extract level).

Builder 1's ``synth.py`` (trajectory level) and ``extract.py`` were not present when builder 2
built and tested the gate library. This helper follows SPEC section 5.1's set dynamics and
content models and writes what the gates read: ``extract/<cell_id>/extract.csv`` with the 58
columns of SPEC 2.2 computed exactly as 2.2 defines them (persist_side = "t"), the
``sidecar.json`` of SPEC 2.3, and ``cells.csv`` of SPEC 2.7. It never writes a trajectory.
When builder 1's modules land, the tests can be re-pointed at ``synth.py corpus`` +
``extract.py all``; the assertions do not change.

Per-page byte pairs for the ``double``/``decay`` models are not drawn one by one (that is
builder 1's exact path); ``l1`` and ``hamming`` per page are drawn from the exact first two
moments of |a - b| and popcount(a XOR b) over the 256 x 255 unequal byte pairs. The
``counter``, ``spin`` and ``idle`` models draw their few bytes exactly.

Builder 2 additions to SynthSpec (test cases of SPEC 5.3 that the SPEC assigns to synth.py flags):
``step`` (--step), ``k_decay`` (--k-decay), ``l0_trend`` (the idle ``trend = -0.5`` on l0 case of
G-DEC), ``role``/``campaign``/``rep``/``archetype`` (cells.csv fields), ``failed_count`` and
``status`` (sidecar fields the precondition tests need).
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

N_DEFAULT = 262144
_Q = (0.05, 0.25, 0.50, 0.75, 0.95)
_QN = ("q05", "q25", "q50", "q75", "q95")

EXTRACT_COLUMNS = tuple(
    ["seq", "K", "n_persist", "n_union", "J", "J_null_inter", "J_null"]
    + [f"{ch}_{s}_all" for ch in ("ham", "l0", "l1") for s in ("sum", *_QN)]
    + [f"{ch}_{s}_per" for ch in ("ham", "l0", "l1") for s in ("sum", *_QN)]
    + [f"r_l0_{q}_per" for q in _QN]
    + [f"r_l1l0_{q}_per" for q in _QN]
    + [f"r_haml0_{q}_per" for q in _QN]
)

KERNELS = (
    ("gemm", "WORKING-SET"), ("floyd", "WORKING-SET"), ("gibbs", "WORKING-SET"),
    ("nbody", "WORKING-SET"), ("spmm", "WORKING-SET"), ("stencil_jacobi", "WORKING-SET"),
    ("fft", "SCATTER"), ("histogram", "SCATTER"), ("fem_assembly", "SCATTER"),
    ("lexer", "SEQUENTIAL-GROW"), ("rmat_gen", "SEQUENTIAL-GROW"),
    ("bnb_tsp", "FRONTIER-CHURN"),
)
KMAP = dict(KERNELS)


def _byte_pair_moments():
    a = np.arange(256)[:, None]
    b = np.arange(256)[None, :]
    mask = a != b
    d = np.abs(a - b)[mask].astype(float)
    x = (a ^ b)[mask]
    pc = np.array([bin(int(v)).count("1") for v in range(256)])[x].astype(float)
    return d.mean(), d.var(), pc.mean(), pc.var()


M1, V1, M2, V2 = _byte_pair_moments()
POPCOUNT = np.array([bin(v).count("1") for v in range(256)])


@dataclass
class SynthSpec:
    name: str
    seed: int
    n_pairs: int = 120
    N: int = N_DEFAULT
    K0: int = 2048
    k_noise: float = 0.02
    churn: float = 0.05
    pulse_period: int | None = None
    pulse_extra: int = 0
    content: str = "double"
    counter_step_max: int = 60
    spin_bytes: tuple = (2, 8)
    double_words: int = 64
    decay_factor: float = 0.7
    floor_F: int = 150
    floor_churn: float = 0.02
    trend: float = 0.0
    seq_first: int = 1
    gap_seqs: tuple = ()
    label: str = "synth"
    # builder 2 additions
    step: bool = False
    step_mode: str = "half"          # "half": K0 then 3 K0 (SPEC --step); "middle": K0, 3 K0, K0 by thirds (no linear drift)
    k_sine_period: int | None = None # slow multiplicative modulation of K (autocorrelated, for the G-ORD resolution case)
    k_sine_amp: float = 0.0
    k_sine_shuffle: bool = False     # the same sinusoid values in a random order (identical marginal, no autocorrelation)
    k_decay: bool = False
    k_decay_factor: float = 0.5
    l0_trend: float = 0.0
    idle_l0_max: int = 4
    role: str = "kernel"
    campaign: str = "01c"
    rep: int = 0
    archetype: str | None = None
    failed_count: int | None = 0
    status: str = "ok"               # the extractor's sidecar status
    cells_status: str = "ok"         # the index's cells.csv status
    floor_noise: float = 0.0         # relative jitter of the floor set size (builder 2 addition)
    cell_id: str | None = None


def _fresh(rng, N, n, exclude):
    """n fresh pages uniform over 0..N-1 not in `exclude`."""
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    out = []
    need = n
    ex = set(exclude.tolist()) if len(exclude) else set()
    while need > 0:
        cand = rng.integers(0, N, size=need * 2 + 8)
        cand = np.unique(cand)
        cand = np.array([c for c in cand if c not in ex], dtype=np.int64)
        take = cand[:need]
        out.append(take)
        ex.update(take.tolist())
        need -= len(take)
    return np.concatenate(out)[:n]


def _evolve(rng, N, cur, target, churn, others):
    """Replace a churn fraction with fresh pages, then resize to `target`."""
    cur = np.asarray(cur, dtype=np.int64)
    n_churn = int(round(churn * len(cur)))
    if n_churn > 0 and len(cur) > 0:
        keep = rng.permutation(len(cur))[n_churn:]
        cur = cur[keep]
    all_ex = np.concatenate([cur, others]) if len(others) else cur
    if len(cur) < target:
        cur = np.concatenate([cur, _fresh(rng, N, target - len(cur), all_ex)])
    elif len(cur) > target:
        cur = cur[rng.permutation(len(cur))[:target]]
    return np.sort(cur)


def _content_page_values(rng, spec, n, model, phase_k, t):
    """Per-page (l0, l1, hamming) for n pages under `model`."""
    if n == 0:
        return (np.zeros(0, dtype=np.int64),) * 3
    if model in ("counter", "idle", "spin"):
        if model == "counter":
            l0 = rng.integers(1, 3, size=n)
        elif model == "idle":
            l0 = rng.integers(1, spec.idle_l0_max + 1, size=n)
        else:
            lo, hi = spec.spin_bytes
            l0 = rng.integers(lo, hi + 1, size=n)
        if spec.l0_trend != 0.0:
            l0 = np.maximum(1, np.round(l0 * (1.0 + spec.l0_trend * t / spec.n_pairs))).astype(np.int64)
        L = int(l0.max())
        mask = np.arange(L)[None, :] < l0[:, None]
        if model == "spin":
            a = rng.integers(1, 255, size=(n, L))
            b = a + rng.choice([-1, 1], size=(n, L))
        else:
            step_max = spec.counter_step_max if model == "counter" else 8
            a = rng.integers(0, 256, size=(n, L))
            b = (a + rng.integers(1, step_max + 1, size=(n, L))) % 256
        l1 = (np.abs(a - b) * mask).sum(axis=1).astype(np.int64)
        ham = (POPCOUNT[a ^ b] * mask).sum(axis=1).astype(np.int64)
        return l0, l1, ham
    if model in ("double", "decay"):
        dw = spec.double_words
        if model == "decay":
            dw = spec.double_words * (spec.decay_factor ** phase_k)
        words = rng.poisson(max(dw, 1e-6), size=n)
        l0 = np.clip(8 * words, 8, 4096).astype(np.int64)
        z1 = rng.standard_normal(n)
        z2 = rng.standard_normal(n)
        l1 = np.clip(np.round(l0 * M1 + np.sqrt(l0 * V1) * z1), l0, 255 * l0).astype(np.int64)
        ham = np.clip(np.round(l0 * M2 + np.sqrt(l0 * V2) * z2), l0, 8 * l0).astype(np.int64)
        return l0, l1, ham
    raise ValueError(model)


def _fmt(x):
    if x is None:
        return ""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, float) and math.isnan(x):
        return ""
    return format(float(x), ".10g")


def _qs(v):
    return [float(q) for q in np.quantile(v, _Q, method="linear")] if len(v) else [None] * 5


def simulate(spec: SynthSpec):
    """Return (rows, truth): rows = list of dicts (one per seq), truth = dict."""
    rng = np.random.default_rng(spec.seed)
    N = spec.N
    n = spec.n_pairs
    work = np.zeros(0, dtype=np.int64)
    floor = np.zeros(0, dtype=np.int64)
    snaps = []  # (pages sorted, l0, l1, ham) aligned
    boundaries = []
    last_b = 0
    sine = None
    if spec.k_sine_period:
        sine = 1.0 + spec.k_sine_amp * np.sin(2.0 * np.pi * np.arange(n) / spec.k_sine_period)
        if spec.k_sine_shuffle:
            sine = sine[rng.permutation(n)]
    for t in range(n):
        seq = spec.seq_first + t
        is_boundary = bool(spec.pulse_period) and t > 0 and (t % spec.pulse_period == 0)
        if is_boundary:
            boundaries.append(seq)
            last_b = t
        phase_k = t - last_b
        # target working-set size
        base = spec.K0 * (1.0 + spec.trend * t / n)
        if spec.step:
            if spec.step_mode == "half":
                base = spec.K0 * (3.0 if t >= n // 2 else 1.0)
            else:
                base = spec.K0 * (3.0 if n // 3 <= t < 2 * n // 3 else 1.0)
        if sine is not None:
            base = base * sine[t]
        if spec.k_decay and spec.pulse_period:
            base = base * (spec.k_decay_factor ** phase_k)
        jitter = 1.0 + rng.uniform(-spec.k_noise, spec.k_noise)
        K_target = max(0, int(round(base * jitter)))
        f_target = spec.floor_F if spec.floor_noise == 0.0 else max(0, int(round(spec.floor_F * (1.0 + rng.uniform(-spec.floor_noise, spec.floor_noise)))))
        floor = _evolve(rng, N, floor, f_target, spec.floor_churn, work)
        work = _evolve(rng, N, work, K_target, spec.churn, floor)
        pulse = _fresh(rng, N, spec.pulse_extra, np.concatenate([work, floor])) if is_boundary else np.zeros(0, dtype=np.int64)
        model = spec.content
        l0w, l1w, hw = _content_page_values(rng, spec, len(work), model, phase_k, t)
        l0p, l1p, hp = _content_page_values(rng, spec, len(pulse), "double" if model in ("counter", "spin", "idle") else model, 0, t)
        l0f, l1f, hf = _content_page_values(rng, spec, len(floor), "idle", 0, t)
        pages = np.concatenate([work, pulse, floor])
        l0 = np.concatenate([l0w, l0p, l0f])
        l1 = np.concatenate([l1w, l1p, l1f])
        ham = np.concatenate([hw, hp, hf])
        order = np.argsort(pages, kind="stable")
        pages, l0, l1, ham = pages[order], l0[order], l1[order], ham[order]
        if seq in spec.gap_seqs:
            pages = np.zeros(0, dtype=np.int64)
            l0 = l1 = ham = np.zeros(0, dtype=np.int64)
        snaps.append((pages, l0, l1, ham))

    rows = []
    for t in range(n):
        seq = spec.seq_first + t
        pages, l0, l1, ham = snaps[t]
        K = len(pages)
        row = {"seq": seq, "K": K}
        for ch, v in (("ham", ham), ("l0", l0), ("l1", l1)):
            row[f"{ch}_sum_all"] = int(v.sum()) if K else 0
            qs = _qs(v) if K else [None] * 5
            for qn, qv in zip(_QN, qs):
                row[f"{ch}_{qn}_all"] = qv
        if t == n - 1:
            for c in EXTRACT_COLUMNS:
                if c not in row:
                    row[c] = None
            rows.append(row)
            continue
        nxt = snaps[t + 1][0]
        K2 = len(nxt)
        P, ia, ib = np.intersect1d(pages, nxt, assume_unique=True, return_indices=True)
        n_persist = len(P)
        n_union = K + K2 - n_persist
        if K == 0 and K2 == 0:
            J = None
        elif K == 0 or K2 == 0:
            J = 0.0
        else:
            J = n_persist / n_union
        jni = K * K2 / N
        den = K + K2 - jni
        jn = jni / den if den > 0 else 0.0
        row.update({"n_persist": n_persist, "n_union": n_union, "J": J, "J_null_inter": jni, "J_null": jn})
        if n_persist == 0:
            for ch in ("ham", "l0", "l1"):
                row[f"{ch}_sum_per"] = None
                for qn in _QN:
                    row[f"{ch}_{qn}_per"] = None
            for r in ("r_l0", "r_l1l0", "r_haml0"):
                for qn in _QN:
                    row[f"{r}_{qn}_per"] = None
        else:
            pl0, pl1, pham = l0[ia], l1[ia], ham[ia]
            for ch, v in (("ham", pham), ("l0", pl0), ("l1", pl1)):
                row[f"{ch}_sum_per"] = int(v.sum())
                for qn, qv in zip(_QN, _qs(v)):
                    row[f"{ch}_{qn}_per"] = qv
            ok = pl0 >= 1
            for r, v in (("r_l0", pl0[ok] / 4096.0), ("r_l1l0", pl1[ok] / pl0[ok]), ("r_haml0", pham[ok] / pl0[ok])):
                for qn, qv in zip(_QN, _qs(v)):
                    row[f"{r}_{qn}_per"] = qv
        rows.append(row)
    truth = {
        "schema": "plan11.synth_truth.v1", "spec": asdict(spec), "boundaries": boundaries,
        "K": [r["K"] for r in rows], "J": [r["J"] for r in rows],
    }
    return rows, truth


def cell_id_of(spec: SynthSpec) -> str:
    if spec.cell_id:
        return spec.cell_id
    k = "idle" if spec.role == "idle" else spec.name
    return f"{k}__rep{spec.rep:02d}__{spec.campaign}"


def write_cell(spec: SynthSpec, out: Path) -> dict:
    """Write extract/<cell_id>/{extract.csv,sidecar.json,truth.json}; return the cells.csv row."""
    out = Path(out)
    cid = cell_id_of(spec)
    d = out / "extract" / cid
    d.mkdir(parents=True, exist_ok=True)
    rows, truth = simulate(spec)
    with (d / "extract.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(EXTRACT_COLUMNS)
        for r in rows:
            w.writerow([_fmt(r[c]) for c in EXTRACT_COLUMNS])
    Ks = np.array([r["K"] for r in rows])
    n_pairs = spec.n_pairs
    kernel = "idle" if spec.role == "idle" else spec.name
    arche = spec.archetype or ("IDLE" if spec.role == "idle" else KMAP.get(spec.name, "unknown"))
    side = {
        "schema": "plan11.extract.v1", "extractor_version": "0.1.0-synth_b2",
        "cell_id": cid, "kernel": kernel, "role": spec.role,
        "archetype_predicted": arche, "seed": spec.seed, "rep": spec.rep, "rep_dir": spec.rep + 1,
        "label": spec.label, "campaign": spec.campaign,
        "path": str(d), "traj_file": f"run_matrix_test1_kernel_{spec.name}_v2.npy.substrate_trajectory.csv.zst",
        "source_bytes": 0, "N": spec.N, "page_size": 4096, "bits_per_page": 32768,
        "duration_s_declared": 600, "quantiles": list(_Q), "persist_side": "t",
        "header_sha256": "synth_b2_header_sha256", "header_ncols": 66,
        "columns_used": {"seq": 0, "page_index": 1, "hamming": 2, "l0": 4, "l1": 5},
        "n_rows_in": int(Ks.sum()), "n_rows_skipped": 0, "n_rows_dup_page": 0,
        "n_rows_zero_hamming": 0, "n_rows_zero_l0": 0,
        "seq_first": spec.seq_first, "seq_last": spec.seq_first + n_pairs - 1,
        "n_seq_present": n_pairs - len(spec.gap_seqs), "n_pairs": n_pairs,
        "n_seq_gaps": len(spec.gap_seqs), "gap_seqs": list(spec.gap_seqs)[:100],
        "dt_est_s": 600.0 / n_pairs, "dt_bracket_s": [0.5, 0.644],
        "K_median": float(np.median(Ks)), "K_max": int(Ks.max()), "apf_max": float(Ks.max() / spec.N),
        "failed_count": spec.failed_count,
        "failed_count_source": "not recorded" if spec.failed_count is None else "--failed-count",
        "status": spec.status,
        "started_at": "2026-09-16T00:00:00Z", "finished_at": "2026-09-16T00:00:01Z", "elapsed_s": 1.0,
    }
    (d / "sidecar.json").write_text(json.dumps(side, indent=1))
    (d / "truth.json").write_text(json.dumps(truth))
    return {
        "cell_id": cid, "kernel": kernel, "role": spec.role, "archetype_predicted": arche,
        "seed": spec.seed, "rep": spec.rep, "rep_dir": spec.rep + 1, "label": spec.label,
        "campaign": spec.campaign, "path": str(d), "traj_file": side["traj_file"],
        "status": spec.cells_status,
    }


CELLS_COLUMNS = ("cell_id", "kernel", "role", "archetype_predicted", "seed", "rep", "rep_dir",
                 "label", "campaign", "path", "traj_file", "status")


def write_corpus(out: Path, specs: list[SynthSpec]) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [write_cell(s, out) for s in specs]
    p = out / "cells.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CELLS_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r[k] is None else r[k]) for k in CELLS_COLUMNS})
    return p


PRESETS = {
    "gemm": dict(K0=4096, pulse_period=24, pulse_extra=4096, churn=0.02, content="double"),
    "floyd": dict(K0=2048, pulse_period=10, pulse_extra=2048, content="decay", decay_factor=0.7),
    "gibbs": dict(K0=256, content="spin", churn=0.03),
    "nbody": dict(K0=2048, content="double", churn=0.01),
    "spmm": dict(K0=1024, content="double", churn=0.05),
    "stencil_jacobi": dict(K0=3072, content="double", churn=0.01),
    "fft": dict(K0=4096, content="double", churn=0.01),
    "histogram": dict(K0=2048, content="counter", churn=0.01),
    "fem_assembly": dict(K0=8256, content="double", churn=0.10),
    "lexer": dict(K0=0, floor_F=150, content="idle"),
    "rmat_gen": dict(K0=512, content="double", churn=0.30, trend=0.5),
    "bnb_tsp": dict(K0=4500, content="double", churn=0.15, k_noise=0.4),
}
IDLE_PRESET = dict(K0=0, floor_F=150, floor_churn=0.02, content="idle", label="idle")


def rep_seed(kernel_index: int, rep: int) -> int:
    return 42 if rep == 0 else 1000 * rep + 100 * (kernel_index + 1)


def corpus_specs(reps: int = 8, idle: int = 8, n_pairs: int = 120, *, seed: int = 20260916,
                 break_pulse=False, break_order=False, cv_case="none", level_only=False,
                 one_preset=False, ord_case=False, kernels=None, campaign_of=None,
                 overrides=None, idle_overrides=None, archetype_consistent=False,
                 campaign_overrides=None) -> list[SynthSpec]:
    """The corpus of SPEC 5.2 with its variants. `campaign_of(kernel, rep) -> str` labels cells;
    `overrides` is {kernel: dict} applied last; `cv_case`: "none" (preset k_noise) | "random" |
    "shot"."""
    rng = np.random.default_rng(seed)
    kernels = list(kernels) if kernels else [k for k, _ in KERNELS]
    specs = []
    for ki, k in enumerate(kernels):
        p = dict(PRESETS[k])
        if level_only:
            p = dict(K0={"WORKING-SET": 2048, "SCATTER": 4096, "SEQUENTIAL-GROW": 8192,
                         "FRONTIER-CHURN": 16384}[KMAP[k]], churn=0.02, content="double")
        if one_preset:
            p = dict(K0=2048, churn=0.02, content="double")
        if archetype_consistent:
            # builder 2 addition: every kernel of an archetype shares one preset (seed and a small K0
            # offset differ), so an archetype is recoverable under LOKO by construction
            p = {"WORKING-SET": dict(K0=2048 + 64 * ki, churn=0.02, content="double", k_noise=0.02),
                 "SCATTER": dict(K0=2048 + 64 * ki, churn=0.02, content="counter", k_noise=0.10),
                 "SEQUENTIAL-GROW": dict(K0=2048 + 64 * ki, churn=0.02, content="spin", k_noise=0.30),
                 "FRONTIER-CHURN": dict(K0=2048 + 64 * ki, churn=0.02, content="double", k_noise=0.50)}[KMAP[k]]
        if ord_case:
            # two archetypes with one marginal and different autocorrelation: WORKING-SET a slow sinusoid,
            # SCATTER iid jitter of about the same spread (order carries the label; a pulse period does not,
            # because the eight shape features are permutation-invariant within a window)
            if KMAP[k] not in ("WORKING-SET", "SCATTER"):
                continue
            p = dict(K0=2048, churn=0.02, content="double", k_sine_period=30, k_sine_amp=0.25, k_noise=0.02,
                     k_sine_shuffle=(KMAP[k] != "WORKING-SET"))
        if break_pulse and k == "gemm":
            p["pulse_period"] = None
            p["pulse_extra"] = 0
        if break_order:
            if k == "gibbs":
                p = dict(PRESETS["gemm"]); p["pulse_period"] = None; p["pulse_extra"] = 0
            if k == "gemm":
                p = dict(PRESETS["gibbs"]); p["pulse_period"] = 24; p["pulse_extra"] = 4096
        if overrides and k in overrides:
            p.update(overrides[k])
        if cv_case == "random":
            p["k_noise"] = float(rng.uniform(0.01, 0.10))
        elif cv_case == "shot":
            p["k_noise"] = 2.0 / math.sqrt(max(p.get("K0", 1), 1))
        for r in range(reps):
            camp = campaign_of(k, r) if campaign_of else "01c"
            pc = dict(p)
            if campaign_overrides and camp in campaign_overrides:
                pc.update(campaign_overrides[camp])
            np_ = pc.pop("n_pairs", n_pairs)
            specs.append(SynthSpec(name=k, seed=rep_seed(ki, r), n_pairs=np_, rep=r,
                                   campaign=camp, **pc))
    for r in range(idle):
        p = dict(IDLE_PRESET)
        if idle_overrides:
            p.update(idle_overrides(r) if callable(idle_overrides) else idle_overrides)
        specs.append(SynthSpec(name="sleep", seed=rep_seed(99, r) + 7, n_pairs=n_pairs, rep=r,
                               role="idle", campaign="01c", **p))
    return specs
