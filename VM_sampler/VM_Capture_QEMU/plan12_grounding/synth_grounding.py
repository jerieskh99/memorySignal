#!/usr/bin/env python3
"""synth_grounding.py -- the smoke corpus of plan12_grounding (SPEC 8.1): a COPY of
`plan11_encoding_ladder/synth.py`'s writer (its `SynthSpec`, page-set dynamics, content models,
`page_channels`, `_format_rows`, `_build_cell`, `write_cell`, presets and `write_corpus`; copied
2026-09-30, the encoding toolkit's file untouched) with two changes:

  1. the cosine column varies plausibly (the original writes `cosine = 0` on every row, its
     `_format_rows` line 403): per changed page, a cosine DISTANCE d in [0, 1] between the page's
     before and after bytes, from the actual changed byte pairs and the expected square sum of the
     unchanged bytes (uniform bytes: E[u^2] = 255 * 511 / 6); then the differ's two conventions
     (SPEC section 3): with probability `cos_p_zero` the page is all zeros on one side, d = 1; with
     probability `cos_p_same` the change keeps the pattern, d = 0; and any angle under about 0.028
     degrees (d < 1.2e-7) is stored as d = 0, as the differ does;
  2. the idle runs are laid out as the real corpus lays them out, `sleep/sleep/sleep_600/
     rep00N__idle_01c/` (SPEC section 2), and the kernel runs as `kernel/kernel_<name>_v2/<sig>/
     rep001__<label>/`;
  3. the trajectory numbers its pairs from 0, as the real capture does (the original's default
     `seq_first = 1` is the encoding toolkit's own convention); the console's reader adds one, so the
     store's pair index runs 1..n_pairs as it does on the real corpus.

Everything else (the 66-column header, the truth.json record, the presets) is the original's. Writes
under the root it is given (the smoke run uses ~/.cache/plan12/smoke/root); never under
plan11_encoding_ladder/.

  synth_grounding.py corpus --root R [--n-pairs 120] [--reps 8] [--idle 8] [--seed 20260916]
                            [--label grounding_smoke] [--p-zero 0.02] [--p-same 0.05] [--no-compress] [--jobs 1]
  synth_grounding.py cell   --root R --name NAME --seed S [--n-pairs 120] [--k0 2048] ... (the original's options)

No server path appears in this file. No sandbox workload is named.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import contextlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import traceback  # noqa: E402
from dataclasses import asdict, dataclass  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import numpy as np  # noqa: E402

from plan11_encoding_ladder import schema  # noqa: E402  (read only: the header line and the constants)
from plan12_grounding import __version__  # noqa: E402

CITATION_SYNTH = ("plan12_grounding/SPEC.md 8.1 (a copy of plan11_encoding_ladder/synth.py's writer with a varying cosine column); "
                  "plan11 SPEC section 5 (the synthetic generator); the differ's units, metrics/family_a/positional.rs")

CONTENT_MODELS = ("counter", "spin", "double", "decay", "idle")
TRAJ_NAME_FMT = "run_matrix_test1_kernel_{name}_v2.npy.substrate_trajectory.csv"
IDLE_TRAJ_NAME = "run_matrix_test1_sleep_600.npy.substrate_trajectory.csv"
_ZERO_TAIL = ",0" * (schema.HEADER_NCOLS_EXPECTED - 9)   # columns 9..65 are 0 (plan11 SPEC 5.1)
_POPCOUNT_U8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
E_U2 = 255.0 * 511.0 / 6.0        # E[u^2] of a uniform byte: sum_{v=0}^{255} v^2 / 256 = 21717.5
D_STORED_AS_ZERO = 1.2e-7         # cos(0.028 degrees) = 1 - 1.19e-7: smaller distances are stored as 0 (SPEC section 3)


# ---------------------------------------------------------------------------
# The spec of one cell (copied; two fields added: param_sig_override, traj_name, cos_p_zero, cos_p_same)
# ---------------------------------------------------------------------------
@dataclass
class SynthSpec:
    name: str                       # kernel name used in the path, e.g. "gemm"
    seed: int
    n_pairs: int = 120
    N: int = 262144
    K0: int = 2048                  # working-set size (breadth level)
    k_noise: float = 0.02           # per-snapshot relative jitter of K (uniform)
    churn: float = 0.05             # fraction of the changed set replaced by fresh pages each snapshot
    pulse_period: int | None = None  # pass length in pairs; None = no pulse
    pulse_extra: int = 0            # extra pages A lit at a boundary (K jumps by pulse_extra)
    content: str = "double"         # "counter" | "spin" | "double" | "decay" | "idle"
    counter_step_max: int = 60
    spin_bytes: tuple[int, int] = (2, 8)
    double_words: int = 64
    decay_factor: float = 0.7
    floor_F: int = 150              # idle-like fixed set size added to every snapshot (0 = none)
    floor_churn: float = 0.02
    trend: float = 0.0
    seq_first: int = 0              # plan12: the real capture numbers pairs from 0 (plan10 runner/trajectory.py)
    gap_seqs: tuple[int, ...] = ()
    label: str = "grounding_smoke"
    step_factor: float = 1.0
    k_decay_factor: float | None = None
    l0_trend: float = 0.0
    rep_dir: int = 1                # the rep<NNN> directory counter
    row_order: str = "ascending"
    pulse_burst: bool = False
    duration_s: float | None = None
    family: str = "kernel"          # the family directory component of the cell path
    test_label_fmt: str = "kernel_{name}_v2"
    # --- plan12 additions ---
    param_sig_override: str | None = None    # the real idle layout uses `sleep_600` where a kernel has its argument signature
    traj_name: str | None = None             # the trajectory file name (default TRAJ_NAME_FMT)
    cos_p_zero: float = 0.02                 # probability a changed page is all zeros on one side (d = 1)
    cos_p_same: float = 0.05                 # probability a change keeps the pattern (d = 0)

    @property
    def effective_duration_s(self) -> float:
        return float(self.duration_s) if self.duration_s is not None else float(self.n_pairs) * float(schema.DT_BRACKET_S[1])

    def validate(self) -> None:
        if self.content not in CONTENT_MODELS:
            raise ValueError(f"content must be one of {CONTENT_MODELS}, got {self.content!r}")
        if self.n_pairs < 1:
            raise ValueError("n_pairs must be >= 1")
        if self.pulse_period is not None and self.pulse_period < 1:
            raise ValueError("pulse_period must be >= 1 or None")
        if self.row_order not in ("ascending", "random"):
            raise ValueError("row_order must be 'ascending' or 'random'")
        if self.K0 < 0 or self.floor_F < 0 or self.pulse_extra < 0:
            raise ValueError("K0, floor_F and pulse_extra must be >= 0")
        if self.K0 + self.floor_F + self.pulse_extra > self.N:
            raise ValueError("K0 + floor_F + pulse_extra exceeds N")
        if not (0.0 <= self.cos_p_zero <= 1.0 and 0.0 <= self.cos_p_same <= 1.0 and self.cos_p_zero + self.cos_p_same <= 1.0):
            raise ValueError("cos_p_zero and cos_p_same must be probabilities summing to at most 1")

    @property
    def test_label(self) -> str:
        return str(self.test_label_fmt).format(name=self.name)

    @property
    def param_sig(self) -> str:
        if self.param_sig_override:
            return self.param_sig_override
        return f"--seed_{self.seed}_--duration_{self.n_pairs}"

    @property
    def rep_component(self) -> str:
        return f"rep{int(self.rep_dir):03d}__{self.label}"

    @property
    def trajectory_name(self) -> str:
        return self.traj_name or TRAJ_NAME_FMT.format(name=self.name)

    def cell_dir(self, root) -> Path:
        return Path(root) / str(self.family) / self.test_label / self.param_sig / self.rep_component


# ---------------------------------------------------------------------------
# Page-set dynamics (copied)
# ---------------------------------------------------------------------------
class _PageSets:
    def __init__(self, N: int, rng: np.random.Generator):
        self.N = int(N)
        self.rng = rng
        self.occupied = np.zeros(self.N, dtype=bool)
        self.W = np.empty(0, dtype=np.int64)
        self.F = np.empty(0, dtype=np.int64)

    def fresh(self, k: int) -> np.ndarray:
        k = int(k)
        if k <= 0:
            return np.empty(0, dtype=np.int64)
        out: list[np.ndarray] = []
        need = k
        while need > 0:
            cand = self.rng.integers(0, self.N, size=int(need * 1.2) + 16, dtype=np.int64)
            cand = np.unique(cand)
            cand = cand[~self.occupied[cand]]
            self.rng.shuffle(cand)
            take = cand[:need]
            self.occupied[take] = True
            out.append(take)
            need -= take.shape[0]
        return np.concatenate(out) if len(out) > 1 else out[0]

    def _drop(self, arr: np.ndarray, n: int) -> np.ndarray:
        n = int(n)
        if n <= 0 or arr.shape[0] == 0:
            return arr
        n = min(n, arr.shape[0])
        idx = self.rng.choice(arr.shape[0], size=n, replace=False)
        self.occupied[arr[idx]] = False
        keep = np.ones(arr.shape[0], dtype=bool)
        keep[idx] = False
        return arr[keep]

    def evolve(self, arr: np.ndarray, target: int, churn: float) -> np.ndarray:
        target = max(0, int(target))
        n_churn = int(round(churn * arr.shape[0]))
        arr = self._drop(arr, n_churn)
        if arr.shape[0] < target:
            arr = np.concatenate([arr, self.fresh(target - arr.shape[0])])
        elif arr.shape[0] > target:
            arr = self._drop(arr, arr.shape[0] - target)
        return arr

    def release(self, arr: np.ndarray) -> None:
        self.occupied[arr] = False


# ---------------------------------------------------------------------------
# Content models (copied)
# ---------------------------------------------------------------------------
def _l0_per_page(spec: SynthSpec, K: int, phase: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    c = spec.content
    if K == 0:
        return np.empty(0, dtype=np.int64)
    if c == "counter":
        hi = max(1, int(round(2 * scale)))
        return rng.integers(1, hi + 1, size=K, dtype=np.int64)
    if c == "idle":
        hi = max(1, int(round(4 * scale)))
        return rng.integers(1, hi + 1, size=K, dtype=np.int64)
    if c == "spin":
        lo, hi = spec.spin_bytes
        hi = max(int(lo), int(round(int(hi) * scale)))
        return rng.integers(int(lo), hi + 1, size=K, dtype=np.int64)
    if c in ("double", "decay"):
        mean_words = float(spec.double_words) * scale
        if c == "decay":
            mean_words *= float(spec.decay_factor) ** int(phase)
        words = rng.poisson(max(mean_words, 1e-9), size=K).astype(np.int64)
        words = np.clip(words, 1, schema.PAGE_SIZE // 8)
        return 8 * words
    raise ValueError(c)


def _uniform_bytes(rng: np.random.Generator, T: int) -> np.ndarray:
    if T <= 0:
        return np.empty(0, dtype=np.uint8)
    return np.frombuffer(rng.bytes(int(T)), dtype=np.uint8).copy()


def _byte_pairs(spec: SynthSpec, T: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    a = _uniform_bytes(rng, T)
    c = spec.content
    if c in ("counter", "idle"):
        step_max = int(spec.counter_step_max) if c == "counter" else 8
        d = rng.integers(1, step_max + 1, size=T, dtype=np.uint8)
        b = a + d
    elif c == "spin":
        up = rng.integers(0, 2, size=T, dtype=np.uint8).astype(bool)
        a16 = a.astype(np.int16)
        b16 = np.where(up, a16 + 1, a16 - 1)
        b16 = np.where(a16 == 255, 254, b16)
        b16 = np.where(a16 == 0, 1, b16)
        b = b16.astype(np.uint8)
    else:  # double, decay
        b = _uniform_bytes(rng, T)
        same = b == a
        n_same = int(same.sum())
        if n_same:
            b[same] = b[same] + rng.integers(1, 256, size=n_same, dtype=np.uint8)
    return a, b


def page_channels(spec: SynthSpec, K: int, phase: int, scale: float, rng: np.random.Generator) -> dict:
    """The differ's positional channels for `K` changed pages (copied) plus, new here, the cosine
    distance `cos` per page (see the module docstring)."""
    l0 = _l0_per_page(spec, K, phase, scale, rng)
    if K == 0:
        z = np.empty(0, dtype=np.int64)
        return {"l0": z, "l1": z, "ham": z, "l2": np.empty(0), "linf": z, "cos": np.empty(0)}
    T = int(l0.sum())
    a, b = _byte_pairs(spec, T, rng)
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    pop = _POPCOUNT_U8[np.bitwise_xor(a, b)]
    starts = np.concatenate([[0], np.cumsum(l0)[:-1]])
    l1 = np.add.reduceat(diff, starts, dtype=np.int64)
    ham = np.add.reduceat(pop, starts, dtype=np.int64)
    linf = np.maximum.reduceat(diff, starts).astype(np.int64)
    d32 = diff.astype(np.int32)
    l2 = np.sqrt(np.add.reduceat(d32 * d32, starts, dtype=np.int64).astype(np.float64))
    # --- the cosine distance of the whole page (plan12): the changed bytes are known, the unchanged
    #     bytes are taken at their expected square sum, so cos = (U E[u^2] + a.b) / (|a'| |b'|)
    a64, b64 = a.astype(np.int64), b.astype(np.int64)
    ab = np.add.reduceat(a64 * b64, starts, dtype=np.int64).astype(np.float64)
    a2 = np.add.reduceat(a64 * a64, starts, dtype=np.int64).astype(np.float64)
    b2 = np.add.reduceat(b64 * b64, starts, dtype=np.int64).astype(np.float64)
    U = (schema.PAGE_SIZE - l0).astype(np.float64)
    na = np.sqrt(np.maximum(U * E_U2 + a2, 1e-12))
    nb = np.sqrt(np.maximum(U * E_U2 + b2, 1e-12))
    cosd = np.clip(1.0 - (U * E_U2 + ab) / (na * nb), 0.0, 1.0)
    u = rng.random(K)
    cosd[u < spec.cos_p_zero] = 1.0                                        # a page all zeros on one side: d = 1 (theta = pi/2)
    cosd[(u >= spec.cos_p_zero) & (u < spec.cos_p_zero + spec.cos_p_same)] = 0.0   # the same pattern, shifted: d = 0
    cosd[cosd < D_STORED_AS_ZERO] = 0.0                                    # angles under about 0.028 degrees are stored as 0
    return {"l0": l0, "l1": l1, "ham": ham, "l2": l2, "linf": linf, "cos": cosd}


# ---------------------------------------------------------------------------
# Truth (copied): every extract column recomputed independently of extract.py
# ---------------------------------------------------------------------------
def _q(arr, qs) -> list[float]:
    return [float(v) for v in np.quantile(np.asarray(arr, dtype=np.float64), qs)]


def truth_row(prev: dict, cur, *, N: int, side: str, qs=schema.QUANTILES) -> dict:
    r: dict = {"seq": prev["seq"], "K": int(prev["pages"].shape[0])}
    K = r["K"]
    n_persist = 0
    if cur is None:
        r.update({"n_persist": None, "n_union": None, "J": None, "J_null_inter": None, "J_null": None})
    else:
        Kc = int(cur["pages"].shape[0])
        inter = set(prev["pages"].tolist()) & set(cur["pages"].tolist())
        n_persist = len(inter)
        n_union = K + Kc - n_persist
        if K == 0 and Kc == 0:
            J = None
        elif K == 0 or Kc == 0:
            J = 0.0
        else:
            J = n_persist / n_union
        jni = K * Kc / N
        den = K + Kc - jni
        r.update({"n_persist": n_persist, "n_union": n_union, "J": J,
                  "J_null_inter": jni, "J_null": (jni / den) if den != 0 else 0.0})
    for ch in schema.CHANNELS:
        arr = prev[ch]
        r[f"{ch}_sum_all"] = int(arr.sum()) if K else 0
        vals = _q(arr, qs) if K else [None] * len(qs)
        for tag, v in zip(schema.QUANTILE_TAGS, vals):
            r[f"{ch}_{tag}_all"] = v
    if cur is None or n_persist == 0:
        for ch in schema.CHANNELS:
            r[f"{ch}_sum_per"] = None
            for tag in schema.QUANTILE_TAGS:
                r[f"{ch}_{tag}_per"] = None
        for col in schema.RATIO_COLUMNS:
            r[col] = None
        return r
    if side == "t":
        mask = np.isin(prev["pages"], cur["pages"])
        src = prev
    else:
        mask = np.isin(cur["pages"], prev["pages"])
        src = cur
    ham, l0, l1 = src["ham"][mask], src["l0"][mask], src["l1"][mask]
    for ch, arr in zip(schema.CHANNELS, (ham, l0, l1)):
        r[f"{ch}_sum_per"] = int(arr.sum())
        for tag, v in zip(schema.QUANTILE_TAGS, _q(arr, qs)):
            r[f"{ch}_{tag}_per"] = v
    for tag, v in zip(schema.QUANTILE_TAGS, _q(l0 / float(schema.PAGE_SIZE), qs)):
        r[f"r_l0_{tag}_per"] = v
    ok = l0 >= 1
    if ok.any():
        l0f = l0[ok].astype(np.float64)
        for tag, v in zip(schema.QUANTILE_TAGS, _q(l1[ok] / l0f, qs)):
            r[f"r_l1l0_{tag}_per"] = v
        for tag, v in zip(schema.QUANTILE_TAGS, _q(ham[ok] / l0f, qs)):
            r[f"r_haml0_{tag}_per"] = v
    else:
        for tag in schema.QUANTILE_TAGS:
            r[f"r_l1l0_{tag}_per"] = None
            r[f"r_haml0_{tag}_per"] = None
    return r


# ---------------------------------------------------------------------------
# Writing the trajectory (copied; _format_rows writes the cosine column)
# ---------------------------------------------------------------------------
def zstd_available() -> str | None:
    if shutil.which("zstd"):
        return "binary"
    try:
        import zstandard  # noqa: F401
        return "module"
    except ImportError:
        return None


@contextlib.contextmanager
def _traj_writer(path: Path, mode: str | None):
    if mode is None:
        with open(path, "w", newline="") as fh:
            yield fh
        return
    if mode == "binary":
        proc = subprocess.Popen(["zstd", "-3", "-q", "-f", "-o", str(path)], stdin=subprocess.PIPE)
        w = io.TextIOWrapper(proc.stdin, encoding="utf-8")
        try:
            yield w
        finally:
            w.flush()
            w.close()
            if proc.wait() != 0:
                raise RuntimeError(f"zstd compress failed for {path} (rc={proc.returncode})")
        return
    import zstandard  # type: ignore
    with open(path, "wb") as raw:
        cctx = zstandard.ZstdCompressor(level=3)
        with cctx.stream_writer(raw) as zw:
            w = io.TextIOWrapper(zw, encoding="utf-8")
            try:
                yield w
            finally:
                w.flush()
                w.detach()


def _format_rows(seq: int, pages: np.ndarray, ch: dict) -> str:
    """The trajectory rows of one snapshot: seq,page_index,hamming,cosine,l0,l1,l2,linf,mean_abs
    then 57 zero columns. The original wrote `0` for cosine; here it is the page's cosine distance."""
    if pages.shape[0] == 0:
        return ""
    l0 = ch["l0"]; l1 = ch["l1"]; ham = ch["ham"]; l2 = ch["l2"]; linf = ch["linf"]; cosd = ch["cos"]
    parts = []
    for i in range(pages.shape[0]):
        l1i = int(l1[i])
        parts.append(f"{seq},{int(pages[i])},{int(ham[i])},{format(float(cosd[i]), '.6g')},{int(l0[i])},{l1i},"
                     f"{format(float(l2[i]), '.6g')},{int(linf[i])},{format(l1i / 4096.0, '.6g')}{_ZERO_TAIL}")
    return "\n".join(parts) + "\n"


def pulse_snapshots(spec: SynthSpec) -> set[int]:
    P = spec.pulse_period
    n = int(spec.n_pairs)
    if not P:
        return set()
    if not spec.pulse_burst:
        return set(range(int(P), n, int(P)))
    out: set[int] = set()
    t = 2 * int(P)
    while t < n:
        out.add(t)
        if t + 1 < n:
            out.add(t + 1)
        t += 2 * int(P)
    return out


def _build_cell(spec: SynthSpec, keep_sets: bool, truth_sides: tuple[str, ...] = ("t",)) -> tuple[list[tuple[int, str]], dict]:
    spec.validate()
    rng = np.random.default_rng(int(spec.seed))
    sets = _PageSets(spec.N, rng)
    n = int(spec.n_pairs)
    gap = set(int(g) for g in spec.gap_seqs)
    blocks: list[tuple[int, str]] = []
    seqs: list[int] = []
    Ks: list[int] = []
    boundaries: list[int] = []
    set_lists: list[list[int]] | None = [] if keep_sets else None
    truth_t: list[dict] = []
    truth_t1: list[dict] = []
    prev: dict | None = None
    n_rows = 0
    n_d1 = n_d0 = 0
    period = spec.pulse_period
    bset = pulse_snapshots(spec)
    for t in range(n):
        seq = int(spec.seq_first) + t
        phase = (t % period) if period else t
        is_boundary = t in bset
        k_target = float(spec.K0) * (1.0 + float(spec.trend) * t / n)
        if spec.step_factor != 1.0 and t >= n / 2.0:
            k_target *= float(spec.step_factor)
        if spec.k_decay_factor is not None:
            k_target *= float(spec.k_decay_factor) ** int(phase)
        jitter = rng.uniform(-float(spec.k_noise), float(spec.k_noise)) if spec.k_noise > 0 else 0.0
        k_target = max(0, int(round(k_target * (1.0 + jitter))))
        sets.W = sets.evolve(sets.W, k_target, float(spec.churn))
        sets.F = sets.evolve(sets.F, int(spec.floor_F), float(spec.floor_churn))
        A = sets.fresh(int(spec.pulse_extra)) if is_boundary else np.empty(0, dtype=np.int64)
        if is_boundary:
            boundaries.append(seq)
        S = np.unique(np.concatenate([sets.W, A, sets.F]))
        if A.shape[0]:
            sets.release(A)
        S_emit = np.empty(0, dtype=np.int64) if seq in gap else S
        scale = 1.0 + float(spec.l0_trend) * t / n
        ch = page_channels(spec, int(S_emit.shape[0]), int(phase), scale, rng)
        if spec.row_order == "random" and S_emit.shape[0] > 1:
            perm = rng.permutation(S_emit.shape[0])
            text = _format_rows(seq, S_emit[perm], {k: v[perm] for k, v in ch.items()})
        else:
            text = _format_rows(seq, S_emit, ch)
        blocks.append((seq, text))
        n_rows += int(S_emit.shape[0])
        n_d1 += int((ch["cos"] >= 1.0).sum())
        n_d0 += int((ch["cos"] <= 0.0).sum())
        cur = {"seq": seq, "pages": S_emit, "ham": ch["ham"], "l0": ch["l0"], "l1": ch["l1"]}
        seqs.append(seq)
        Ks.append(int(S_emit.shape[0]))
        if set_lists is not None:
            set_lists.append([int(x) for x in S_emit])
        if prev is not None:
            truth_t.append(truth_row(prev, cur, N=spec.N, side="t"))
            if "t+1" in truth_sides:
                truth_t1.append(truth_row(prev, cur, N=spec.N, side="t+1"))
        prev = cur
    assert prev is not None
    truth_t.append(truth_row(prev, None, N=spec.N, side="t"))
    if "t+1" in truth_sides:
        truth_t1.append(truth_row(prev, None, N=spec.N, side="t+1"))
    per_seq = {col: [row[col] for row in truth_t] for col in schema.EXTRACT_COLUMNS}
    per_seq_t1 = ({col: [row[col] for row in truth_t1] for col in schema.EXTRACT_COLUMNS} if "t+1" in truth_sides else None)
    spec_rec = asdict(spec)
    spec_rec["duration_s"] = spec.effective_duration_s
    truth = {
        "schema": "plan12.synth_truth.v1",
        "generator_version": __version__,
        "citation": CITATION_SYNTH,
        "spec": spec_rec,
        "duration_s": spec.effective_duration_s,
        "seq": seqs, "K": Ks,
        "J": per_seq["J"], "n_persist": per_seq["n_persist"],
        "boundaries": boundaries,
        "sets": set_lists,
        "per_seq_channels": per_seq,
        "per_seq_channels_t1": per_seq_t1,
        "persist_side_of_per_seq_channels": "t",
        "cosine": {"rows_at_d1": n_d1, "rows_at_d0": n_d0, "p_zero": spec.cos_p_zero, "p_same": spec.cos_p_same,
                   "stored_as_zero_below": D_STORED_AS_ZERO, "unchanged_bytes_E_u2": E_U2},
        "n_rows": n_rows,
        "header": schema.TRAJ_HEADER_LINE,
    }
    return blocks, truth


def write_cell(spec: SynthSpec, root, *, compress: bool = True, keep_sets: bool | None = None,
               truth_sides: tuple[str, ...] = ("t",)) -> Path:
    spec.validate()
    if keep_sets is None:
        keep_sets = (spec.n_pairs * max(spec.K0, 1) <= 2e6)
    blocks, truth = _build_cell(spec, keep_sets=keep_sets, truth_sides=truth_sides)
    cell = spec.cell_dir(root)
    cell.mkdir(parents=True, exist_ok=True)
    mode = zstd_available() if compress else None
    name = spec.trajectory_name + (".zst" if mode else "")
    path = cell / name
    with _traj_writer(path, mode) as fh:
        fh.write(schema.TRAJ_HEADER_LINE + "\n")
        for _, text in blocks:
            if text:
                fh.write(text)
    truth["traj_file"] = name
    truth["compressed"] = mode
    truth["written_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(cell / "truth.json", "w") as fh:
        json.dump(truth, fh)
    return cell


# ---------------------------------------------------------------------------
# The corpus (the original's presets; the idle runs in the real corpus's layout)
# ---------------------------------------------------------------------------
PRESETS: dict[str, dict] = {
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
IDLE_PRESET: dict = dict(K0=0, floor_F=150, floor_churn=0.02, content="idle")


def rep_seed(rep: int, kernel_index: int) -> int:
    """AA A4: rep 0 is seed 42; reps 1..7 are 1000 * r + base, base = the kernel's index."""
    return schema.REP0_SEED if rep == 0 else 1000 * int(rep) + int(kernel_index)


def corpus_specs(*, n_pairs: int = 120, reps: int = 8, idle: int = 8, seed: int = 20260916, label: str = "grounding_smoke",
                 p_zero: float = 0.02, p_same: float = 0.05, duration_s: float | None = None) -> list[SynthSpec]:
    """12 kernels x `reps` runs from the presets (k_noise per kernel from Uniform(0.01, 0.10) where the
    preset does not set it, as the original's default case) plus `idle` idle runs laid out as
    `sleep/sleep/sleep_600/rep00N__idle_01c/`."""
    rng = np.random.default_rng(int(seed))
    random_noise = {k: float(rng.uniform(0.01, 0.10)) for k in schema.KERNEL_NAMES}
    specs: list[SynthSpec] = []
    for ki, (kernel, _arch) in enumerate(schema.KERNELS):
        p = dict(PRESETS[kernel])
        if "k_noise" not in p:
            p["k_noise"] = random_noise[kernel]
        for r in range(int(reps)):
            specs.append(SynthSpec(name=kernel, seed=rep_seed(r, ki), n_pairs=int(n_pairs), rep_dir=1, label=label,
                                   duration_s=duration_s, cos_p_zero=p_zero, cos_p_same=p_same, **p))
    for r in range(int(idle)):
        specs.append(SynthSpec(name="sleep", seed=rep_seed(r, len(schema.KERNELS)), n_pairs=int(n_pairs),
                               family="sleep", test_label_fmt="sleep", param_sig_override="sleep_600",
                               rep_dir=r + 1, label="idle_01c", traj_name=IDLE_TRAJ_NAME,
                               duration_s=duration_s, cos_p_zero=p_zero, cos_p_same=p_same, **IDLE_PRESET))
    return specs


def _corpus_worker(args: tuple) -> str:
    spec, root, compress, keep_sets = args
    return str(write_cell(spec, root, compress=compress, keep_sets=keep_sets, truth_sides=("t",)))


def write_corpus(root, *, compress: bool = True, keep_sets: bool = False, jobs: int = 1, **kw) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    specs = corpus_specs(**kw)
    tasks = [(s, str(root), compress, keep_sets) for s in specs]
    if int(jobs) > 1:
        import multiprocessing as mp
        order = sorted(range(len(tasks)), key=lambda i: -(specs[i].K0 + specs[i].floor_F + specs[i].pulse_extra))
        paths = [None] * len(tasks)
        with mp.Pool(processes=int(jobs)) as pool:
            for i, path in zip(order, pool.imap(_corpus_worker, [tasks[i] for i in order], chunksize=1)):
                paths[i] = path
    else:
        paths = [_corpus_worker(t) for t in tasks]
    cells = [{"path": p, "spec": asdict(s)} for p, s in zip(paths, specs)]
    manifest = {
        "schema": "plan12.synth_corpus.v1",
        "params": {**{k: (list(v) if isinstance(v, tuple) else v) for k, v in kw.items()},
                   "compress": bool(compress), "keep_sets": bool(keep_sets), "jobs": int(jobs),
                   "presets": PRESETS, "idle_preset": IDLE_PRESET, "idle_layout": "sleep/sleep/sleep_600/rep00N__idle_01c"},
        "citation": CITATION_SYNTH,
        "n_cells": len(cells), "cells": cells,
        "generator_version": __version__,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(root / "corpus.json", "w") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _int_list(s: str | None) -> tuple[int, ...]:
    if not s:
        return ()
    return tuple(int(x) for x in s.split(",") if x.strip())


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="synth_grounding.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cell", help="write one synthetic cell")
    c.add_argument("--root", required=True)
    c.add_argument("--name", required=True)
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--n-pairs", type=int, default=120)
    c.add_argument("--k0", type=int, default=2048)
    c.add_argument("--k-noise", type=float, default=0.02)
    c.add_argument("--churn", type=float, default=0.05)
    c.add_argument("--pulse-period", type=int, default=None)
    c.add_argument("--pulse-extra", type=int, default=0)
    c.add_argument("--content", choices=CONTENT_MODELS, default="double")
    c.add_argument("--floor-f", type=int, default=150)
    c.add_argument("--floor-churn", type=float, default=0.02)
    c.add_argument("--trend", type=float, default=0.0)
    c.add_argument("--seq-first", type=int, default=0)
    c.add_argument("--gap-seqs", default="")
    c.add_argument("--label", default="grounding_smoke")
    c.add_argument("--rep-dir", type=int, default=1)
    c.add_argument("--p-zero", type=float, default=0.02)
    c.add_argument("--p-same", type=float, default=0.05)
    c.add_argument("--no-compress", action="store_true")
    c.add_argument("--duration-s", type=float, default=None)
    k = sub.add_parser("corpus", help="write the smoke corpus: 12 kernels x reps, plus the idle runs")
    k.add_argument("--root", required=True)
    k.add_argument("--n-pairs", type=int, default=120)
    k.add_argument("--reps", type=int, default=8)
    k.add_argument("--idle", type=int, default=8)
    k.add_argument("--seed", type=int, default=20260916)
    k.add_argument("--label", default="grounding_smoke")
    k.add_argument("--p-zero", type=float, default=0.02)
    k.add_argument("--p-same", type=float, default=0.05)
    k.add_argument("--no-compress", action="store_true")
    k.add_argument("--truth-sets", action="store_true")
    k.add_argument("--jobs", type=int, default=1)
    k.add_argument("--duration-s", type=float, default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = _build_parser()
    a = ap.parse_args(argv)
    try:
        if a.cmd == "cell":
            spec = SynthSpec(name=a.name, seed=a.seed, n_pairs=a.n_pairs, K0=a.k0, k_noise=a.k_noise, churn=a.churn,
                             pulse_period=a.pulse_period, pulse_extra=a.pulse_extra, content=a.content,
                             floor_F=a.floor_f, floor_churn=a.floor_churn, trend=a.trend, seq_first=a.seq_first,
                             gap_seqs=_int_list(a.gap_seqs), label=a.label, rep_dir=a.rep_dir,
                             cos_p_zero=a.p_zero, cos_p_same=a.p_same, duration_s=a.duration_s)
            cell = write_cell(spec, a.root, compress=not a.no_compress)
            print(f"[synth_grounding] wrote {cell}")
            return 0
        if a.cmd == "corpus":
            m = write_corpus(a.root, compress=not a.no_compress, keep_sets=a.truth_sets, jobs=a.jobs,
                             n_pairs=a.n_pairs, reps=a.reps, idle=a.idle, seed=a.seed, label=a.label,
                             p_zero=a.p_zero, p_same=a.p_same, duration_s=a.duration_s)
            print(f"[synth_grounding] wrote {m['n_cells']} cells under {a.root}")
            return 0
        ap.error(f"unknown command {a.cmd}")
        return 1
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
