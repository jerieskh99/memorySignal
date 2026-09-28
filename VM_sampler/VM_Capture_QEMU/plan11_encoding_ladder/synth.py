#!/usr/bin/env python3
"""synth.py -- the synthetic trajectory generator and the synthetic corpus (SPEC section 5,
builder 1). Every test in the package uses it; there is no real data on the build machine.

A synthetic cell is a directory laid out like a retention cell
(`<root>/kernel/kernel_<name>_v2/<sig>/rep<NNN>__<label>/`, SPEC 5.1) holding one trajectory
file with the exact 66-column header of SPEC 2.1 and a `truth.json` that records every extract
column recomputed by the generator from its own arrays, so `tests/test_extract.py` can assert
the extractor against known answers (J, the persistent-page ratios, gaps, the last row's blanks).

Subcommands (SPEC 7.1):
  synth.py cell   --root R --name NAME --seed S [--n-pairs 120] [--k0 2048] [--k-noise 0.02]
                  [--churn 0.05] [--pulse-period P] [--pulse-extra A]
                  [--content counter|spin|double|decay|idle] [--counter-step-max 60]
                  [--spin-bytes 2,8] [--double-words 64] [--decay-factor 0.7] [--floor-f 150]
                  [--floor-churn 0.02] [--trend 0.0] [--seq-first 1] [--gap-seqs 5,9]
                  [--label synth] [--no-compress] [--corrupt seq_reverse] [--step] [--k-decay]
                  extensions: [--l0-trend 0.0] [--rep-dir 1] [--row-order ascending|random]
                  [--k-decay-factor 0.5] [--no-truth-sets]
  synth.py corpus --root R [--n-pairs 120] [--reps 8] [--idle 8] [--seed 20260916]
                  [--break-pulse] [--break-order] [--cv-case random|shot|preset] [--level-only]
                  [--one-preset] [--ord-case] [--ord-periods 6,12] [--ord-mode period|burst]
                  [--no-compress] [--truth-sets] [--jobs 1]

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
import math  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import traceback  # noqa: E402
from dataclasses import asdict, dataclass  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import numpy as np  # noqa: E402

from plan11_encoding_ladder import __version__, schema  # noqa: E402

CITATION_SYNTH = "SPEC section 5 (the synthetic generator and corpus); the differ's units, metrics/family_a/positional.rs"

CONTENT_MODELS = ("counter", "spin", "double", "decay", "idle")
CORRUPTIONS = ("seq_reverse",)
TRAJ_NAME_FMT = "run_matrix_test1_kernel_{name}_v2.npy.substrate_trajectory.csv"
_ZERO_TAIL = ",0" * (schema.HEADER_NCOLS_EXPECTED - 9)   # columns 9..65 are 0 (SPEC 5.1)
_POPCOUNT_U8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# ---------------------------------------------------------------------------
# The spec of one cell (SPEC 5.1) with builder 1's documented extensions
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
    counter_step_max: int = 60      # content="counter": |byte difference| uniform on 1..counter_step_max
    spin_bytes: tuple[int, int] = (2, 8)   # content="spin": l0 per page uniform on this range, |diff| = 1
    double_words: int = 64          # content="double"/"decay": mean number of re-randomized 8-byte words per page
    decay_factor: float = 0.7       # per-snapshot multiplier of the median l0 inside a pass (content="decay")
    floor_F: int = 150              # idle-like fixed set size added to every snapshot (0 = none)
    floor_churn: float = 0.02       # churn of the floor set
    trend: float = 0.0              # linear drift of K over the cell, as a fraction of K0
    seq_first: int = 1
    gap_seqs: tuple[int, ...] = ()  # snapshots emitted with zero rows (K = 0)
    label: str = "synth"
    # --- extensions by builder 1 for the cases of SPEC 5.3 (see BUILD_extract.md) ---
    step_factor: float = 1.0        # --step: K0 for the first half of the cell, step_factor * K0 after
    k_decay_factor: float | None = None  # --k-decay: K also shrinks by this factor per snapshot inside a pass
    l0_trend: float = 0.0           # linear drift of the content's l0 scale over the cell (fraction)
    corrupt: str | None = None      # "seq_reverse": two snapshot blocks swapped so seq decreases once
    rep_dir: int = 1                # the rep<NNN> directory counter
    row_order: str = "ascending"    # "ascending" (as the differ) | "random" (exercises the sort)
    pulse_burst: bool = False       # ord-case "burst": pulses fire in adjacent pairs every 2 * pulse_period
    # --- SPEC_epoch2 B12 (CHECK_3 M12): the cell's guest-running seconds, recorded in truth.json so that the
    #     extractor can be run at `--duration-s truth["duration_s"]`; None = n_pairs x 0.644 s (the paper's
    #     median guest spacing, SPEC 2.3 and schema.DT_BRACKET_S[1]), never the real corpus's 600 s
    duration_s: float | None = None
    # --- additive extension for the detection corpus (SPEC_DETECTION.md 1.3 and 4.2) ---
    family: str = "kernel"          # the family directory component of the cell path
    test_label_fmt: str = "kernel_{name}_v2"   # the test-label component, formatted with name=self.name

    @property
    def effective_duration_s(self) -> float:
        """``duration_s`` when declared, else ``n_pairs * schema.DT_BRACKET_S[1]`` (0.644 s per pair;
        SPEC_epoch2 B12, Part 4 item 19)."""
        return float(self.duration_s) if self.duration_s is not None else float(self.n_pairs) * float(schema.DT_BRACKET_S[1])

    def validate(self) -> None:
        if self.content not in CONTENT_MODELS:
            raise ValueError(f"content must be one of {CONTENT_MODELS}, got {self.content!r}")
        if self.corrupt is not None and self.corrupt not in CORRUPTIONS:
            raise ValueError(f"corrupt must be one of {CORRUPTIONS}, got {self.corrupt!r}")
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

    @property
    def test_label(self) -> str:
        return str(self.test_label_fmt).format(name=self.name)

    @property
    def param_sig(self) -> str:
        return f"--seed_{self.seed}_--duration_{self.n_pairs}"

    @property
    def rep_component(self) -> str:
        return f"rep{int(self.rep_dir):03d}__{self.label}"

    def cell_dir(self, root) -> Path:
        return Path(root) / str(self.family) / self.test_label / self.param_sig / self.rep_component


# ---------------------------------------------------------------------------
# Page-set dynamics (SPEC 5.1 "Set dynamics")
# ---------------------------------------------------------------------------
class _PageSets:
    """The working set W, the floor set F and the pulse set A over 0..N-1, with an occupancy
    array so fresh pages are never members of the current set."""

    def __init__(self, N: int, rng: np.random.Generator):
        self.N = int(N)
        self.rng = rng
        self.occupied = np.zeros(self.N, dtype=bool)
        self.W = np.empty(0, dtype=np.int64)
        self.F = np.empty(0, dtype=np.int64)

    def fresh(self, k: int) -> np.ndarray:
        """`k` distinct pages not currently occupied (uniform over the free pages)."""
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
        """Replace a `churn` fraction of `arr` by fresh pages, then resize to `target`."""
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
# Content models (SPEC 5.1 table): per changed page, l0 and actual byte pairs (a, b), a != b
# ---------------------------------------------------------------------------
def _l0_per_page(spec: SynthSpec, K: int, phase: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    """The changed-byte count per page for `K` pages under the content model. `phase` is the
    snapshot's offset inside its pass (decay), `scale` the l0 drift factor (l0_trend)."""
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
    """`T` uniform bytes from the generator's raw stream (`Generator.bytes`), writable uint8."""
    if T <= 0:
        return np.empty(0, dtype=np.uint8)
    return np.frombuffer(rng.bytes(int(T)), dtype=np.uint8).copy()


def _byte_pairs(spec: SynthSpec, T: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """`T` byte pairs (a, b) as uint8 arrays with a != b under the content model (SPEC 5.1)."""
    a = _uniform_bytes(rng, T)
    c = spec.content
    if c in ("counter", "idle"):
        step_max = int(spec.counter_step_max) if c == "counter" else 8
        d = rng.integers(1, step_max + 1, size=T, dtype=np.uint8)
        b = a + d                                   # uint8 wraps: (a + d) mod 256, d >= 1 so b != a
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
            b[same] = b[same] + rng.integers(1, 256, size=n_same, dtype=np.uint8)   # wraps, never equal
    return a, b


def page_channels(spec: SynthSpec, K: int, phase: int, scale: float, rng: np.random.Generator) -> dict:
    """The differ's positional channels for `K` changed pages: l0, l1, hamming, l2, linf,
    from actual byte pairs, so the identities 1 <= l0 <= 4096, l1 >= l0, hamming <= 8 l0,
    linf <= 255, l2 <= l1 and mean_abs = l1 / 4096 hold exactly (SPEC 5.1)."""
    l0 = _l0_per_page(spec, K, phase, scale, rng)
    if K == 0:
        z = np.empty(0, dtype=np.int64)
        return {"l0": z, "l1": z, "ham": z, "l2": np.empty(0), "linf": z}
    T = int(l0.sum())
    a, b = _byte_pairs(spec, T, rng)
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))          # 1..255 per byte
    pop = _POPCOUNT_U8[np.bitwise_xor(a, b)]                          # 1..8 per byte
    starts = np.concatenate([[0], np.cumsum(l0)[:-1]])
    l1 = np.add.reduceat(diff, starts, dtype=np.int64)
    ham = np.add.reduceat(pop, starts, dtype=np.int64)
    linf = np.maximum.reduceat(diff, starts).astype(np.int64)
    d32 = diff.astype(np.int32)
    l2 = np.sqrt(np.add.reduceat(d32 * d32, starts, dtype=np.int64).astype(np.float64))
    return {"l0": l0, "l1": l1, "ham": ham, "l2": l2, "linf": linf}


# ---------------------------------------------------------------------------
# Truth: every extract column recomputed independently of extract.py (SPEC 5.1 truth.json)
# ---------------------------------------------------------------------------
def _q(arr, qs) -> list[float]:
    return [float(v) for v in np.quantile(np.asarray(arr, dtype=np.float64), qs)]


def truth_row(prev: dict, cur, *, N: int, side: str, qs=schema.QUANTILES) -> dict:
    """The extract row of `prev` against `cur` (None at the last seq) by the definitions of
    SPEC 2.2, computed with Python sets and `numpy.isin` rather than the extractor's
    `intersect1d` path. Keys are `schema.EXTRACT_COLUMNS`; blanks are None."""
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
    assert int(mask.sum()) == n_persist
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
# Writing the trajectory
# ---------------------------------------------------------------------------
def zstd_available() -> str | None:
    """"binary" when the `zstd` binary is on PATH, "module" when only `zstandard` imports, else None."""
    if shutil.which("zstd"):
        return "binary"
    try:
        import zstandard  # noqa: F401
        return "module"
    except ImportError:
        return None


@contextlib.contextmanager
def _traj_writer(path: Path, mode: str | None):
    """A text handle writing `path`: through `zstd -3 -q -f -o` (mode "binary"), through
    `zstandard` (mode "module"), or plain (None). The plan08 `field_writer` pattern."""
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
    then 57 zero columns (SPEC 5.1: every other column 0)."""
    if pages.shape[0] == 0:
        return ""
    l0 = ch["l0"]; l1 = ch["l1"]; ham = ch["ham"]; l2 = ch["l2"]; linf = ch["linf"]
    parts = []
    for i in range(pages.shape[0]):
        l1i = int(l1[i])
        parts.append(f"{seq},{int(pages[i])},{int(ham[i])},0,{int(l0[i])},{l1i},"
                     f"{format(float(l2[i]), '.6g')},{int(linf[i])},{format(l1i / 4096.0, '.6g')}{_ZERO_TAIL}")
    return "\n".join(parts) + "\n"


def pulse_snapshots(spec: SynthSpec) -> set[int]:
    """The 0-based snapshots at which the pulse set A is lit (SPEC 5.1: t % pulse_period == 0,
    t > 0). With `pulse_burst` (the ord-case "burst" class) the pulses come in adjacent pairs
    at t = 2 P k and 2 P k + 1 for k >= 1, so a burst cell carries about the same number of
    pulses as a period-P cell (one fewer at n = 120, P = 12) in a different order."""
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


def _build_cell(spec: SynthSpec, keep_sets: bool, truth_sides: tuple[str, ...] = ("t", "t+1")) -> tuple[list[tuple[int, str]], dict]:
    """Run the dynamics (SPEC 5.1) and produce the text block of every snapshot plus the truth.
    Memory: one snapshot's channel arrays at a time plus the previous snapshot (for the truth
    row); the text blocks are kept as strings (a synthetic cell is small). `truth_sides` says
    which persist sides get a full truth table (`per_seq_channels` is always the "t" side;
    `per_seq_channels_t1` is written only when "t+1" is requested)."""
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
    period = spec.pulse_period
    bset = pulse_snapshots(spec)
    for t in range(n):
        seq = int(spec.seq_first) + t
        # --- breadth level for this snapshot ---
        phase = (t % period) if period else t
        is_boundary = t in bset
        k_target = float(spec.K0) * (1.0 + float(spec.trend) * t / n)
        if spec.step_factor != 1.0 and t >= n / 2.0:
            k_target *= float(spec.step_factor)
        if spec.k_decay_factor is not None:
            k_target *= float(spec.k_decay_factor) ** int(phase)
        jitter = rng.uniform(-float(spec.k_noise), float(spec.k_noise)) if spec.k_noise > 0 else 0.0
        k_target = max(0, int(round(k_target * (1.0 + jitter))))
        # --- sets ---
        sets.W = sets.evolve(sets.W, k_target, float(spec.churn))
        sets.F = sets.evolve(sets.F, int(spec.floor_F), float(spec.floor_churn))
        A = sets.fresh(int(spec.pulse_extra)) if is_boundary else np.empty(0, dtype=np.int64)
        if is_boundary:
            boundaries.append(seq)
        S = np.unique(np.concatenate([sets.W, A, sets.F]))
        if A.shape[0]:
            sets.release(A)                # A_t is dropped at t + 1 (SPEC 5.1)
        if seq in gap:
            S_emit = np.empty(0, dtype=np.int64)
        else:
            S_emit = S
        # --- content ---
        scale = 1.0 + float(spec.l0_trend) * t / n
        ch = page_channels(spec, int(S_emit.shape[0]), int(phase), scale, rng)
        if spec.row_order == "random" and S_emit.shape[0] > 1:
            perm = rng.permutation(S_emit.shape[0])
            text = _format_rows(seq, S_emit[perm], {k: v[perm] for k, v in ch.items()})
        else:
            text = _format_rows(seq, S_emit, ch)
        blocks.append((seq, text))
        n_rows += int(S_emit.shape[0])
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
    per_seq_t1 = ({col: [row[col] for row in truth_t1] for col in schema.EXTRACT_COLUMNS}
                  if "t+1" in truth_sides else None)
    F = int(spec.floor_F)
    expected = {
        "J_steady": (1.0 - spec.churn) / (1.0 + spec.churn) if spec.K0 > 0 else None,
        "J_at_boundary": (spec.K0 / (spec.K0 + spec.pulse_extra)) if (spec.pulse_extra and (spec.K0 + spec.pulse_extra)) else None,
        "K_jump_ratio": ((spec.K0 + spec.pulse_extra + F) / (spec.K0 + F)) if (spec.pulse_extra and (spec.K0 + F)) else None,
        "hamming_over_l0": {"counter": "about 2 to 3", "spin": "about 2", "double": "about 4.0", "decay": "about 4.0", "idle": "about 2"}[spec.content],
        "l1_over_l0": {"counter": f"about {(1 + spec.counter_step_max) / 2:.1f} (wraps aside)", "spin": "1", "double": "about 85.7", "decay": "about 85.7", "idle": "about 4.5"}[spec.content],
        "l0_per_page": {"counter": "1 to 2", "spin": f"{spec.spin_bytes[0]} to {spec.spin_bytes[1]}", "double": f"8 x Poisson({spec.double_words})", "decay": f"8 x Poisson({spec.double_words} x {spec.decay_factor}^phase)", "idle": "1 to 4"}[spec.content],
    }
    spec_rec = asdict(spec)
    spec_rec["duration_s"] = spec.effective_duration_s      # SPEC_epoch2 B12: the declared duration, resolved
    truth = {
        "schema": schema.SYNTH_TRUTH_SCHEMA,
        "generator_version": __version__,
        "citation": CITATION_SYNTH,
        "spec": spec_rec,
        "duration_s": spec.effective_duration_s,
        "duration_s_source": "declared" if spec.duration_s is not None else "n_pairs x 0.644 s (SPEC 2.3 median guest spacing)",
        "seq": seqs, "K": Ks,
        "J": per_seq["J"], "n_persist": per_seq["n_persist"],
        "boundaries": boundaries,
        "sets": set_lists,
        "per_seq_channels": per_seq,
        "per_seq_channels_t1": per_seq_t1,
        "persist_side_of_per_seq_channels": "t",
        "expected": expected,
        "n_rows": n_rows,
        "header": schema.TRAJ_HEADER_LINE,
    }
    return blocks, truth


def write_cell(spec: SynthSpec, root, *, compress: bool = True, keep_sets: bool | None = None,
               truth_sides: tuple[str, ...] = ("t", "t+1")) -> Path:
    """Write one synthetic cell (SPEC 5.1): the directory
    `<root>/kernel/kernel_<name>_v2/--seed_<seed>_--duration_<n_pairs>/rep<NNN>__<label>/`, the
    trajectory `run_matrix_test1_kernel_<name>_v2.npy.substrate_trajectory.csv.zst` (plain
    `.csv` when `compress` is False or no zstd is available) with the exact 66-column header,
    and `truth.json`. `keep_sets=None` applies the SPEC rule (sorted page lists kept when
    n_pairs * K0 <= 2e6); True / False force it. With `spec.corrupt == "seq_reverse"` the
    blocks of the third and fourth snapshots are swapped in the file (the truth is unchanged,
    the extractor must refuse). `truth_sides` as in `_build_cell` (the corpus writes the "t"
    side only). Returns the cell directory."""
    spec.validate()
    if keep_sets is None:
        keep_sets = (spec.n_pairs * max(spec.K0, 1) <= 2e6)
    blocks, truth = _build_cell(spec, keep_sets=keep_sets, truth_sides=truth_sides)
    if spec.corrupt == "seq_reverse":
        if len(blocks) < 4:
            raise ValueError("seq_reverse needs at least 4 snapshots")
        i, j = 2, 3
        # the swapped blocks must both carry rows for the reversal to be visible
        while j < len(blocks) and (not blocks[i][1] or not blocks[j][1]):
            i, j = i + 1, j + 1
        if j >= len(blocks):
            raise ValueError("seq_reverse needs two consecutive snapshots with rows")
        blocks[i], blocks[j] = blocks[j], blocks[i]
    cell = spec.cell_dir(root)
    cell.mkdir(parents=True, exist_ok=True)
    mode = zstd_available() if compress else None
    name = TRAJ_NAME_FMT.format(name=spec.name) + (".zst" if mode else "")
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
# The corpus (SPEC 5.2) with the corrections of the two reviews
# ---------------------------------------------------------------------------
# floyd's `pulse_extra = 2048` is al-Kindi's review section 2 item 4 (d): the SPEC's preset had
# no K jump, so G-DEC's k_jump boundary source could never fire on the must-pass case.
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
IDLE_PRESET: dict = dict(K0=0, floor_F=150, floor_churn=0.02, content="idle", label="idle", name="sleep")
_CONTENT_KEYS = ("content", "counter_step_max", "spin_bytes", "double_words", "decay_factor")
LEVEL_ONLY_K0 = {"WORKING-SET": 2048, "SCATTER": 4096, "SEQUENTIAL-GROW": 8192, "FRONTIER-CHURN": 16384}
ONE_PRESET: dict = dict(K0=2048, content="double", churn=0.02)


def rep_seed(rep: int, kernel_index: int) -> int:
    """AA A4: rep 0 is seed 42; reps 1..7 are 1000 * r + base, base = the kernel's index."""
    return schema.REP0_SEED if rep == 0 else 1000 * int(rep) + int(kernel_index)


def corpus_specs(*, n_pairs: int = 120, reps: int = 8, idle: int = 8, seed: int = 20260916,
                 break_pulse: bool = False, break_order: bool = False, cv_case: str = "random",
                 level_only: bool = False, one_preset: bool = False, ord_case: bool = False,
                 ord_periods: tuple[int, int] = (6, 12), ord_mode: str = "period",
                 duration_s: float | None = None) -> list[SynthSpec]:
    """The specs of the synthetic corpus (SPEC 5.2): 12 kernels x `reps` cells plus `idle` idle
    cells from the presets, with the case switches of SPEC 5.2 and 5.3:
    `break_pulse` (gemm without its pulse: G-C must refuse), `break_order` (gibbs's and gemm's
    content presets swapped: the content ordering must fail), `cv_case` ("random": k_noise per
    kernel from Uniform(0.01, 0.10) independent of K0, the corpus seed drawing it, for every
    kernel whose preset does not set k_noise explicitly and not under level_only / one_preset /
    ord_case, where every kernel keeps the preset's k_noise so that only the declared difference
    carries the label; "shot": k_noise = 2 / sqrt(K0), the G-L (ii) refusal case; "preset": the
    table's values), `level_only` (every kernel the gemm
    preset without the pulse and a per-archetype K0 of 2048 / 4096 / 8192 / 16384),
    `one_preset` (every kernel the same preset, only the seed differs), `ord_case` (two
    period classes on identical K0 and content: archetypes at even index in ARCHETYPES[1:] get
    `ord_periods[0]`, odd get `ord_periods[1]`; `ord_mode = "burst"` instead gives both classes
    the same pulse count with the second class firing its pulses in adjacent pairs, so only the
    order differs, see BUILD_extract.md)."""
    if cv_case not in ("random", "shot", "preset"):
        raise ValueError("cv_case must be random, shot or preset")
    if ord_mode not in ("period", "burst"):
        raise ValueError("ord_mode must be period or burst")
    rng = np.random.default_rng(int(seed))
    random_noise = {k: float(rng.uniform(0.01, 0.10)) for k in schema.KERNEL_NAMES}
    specs: list[SynthSpec] = []
    presets = {k: dict(v) for k, v in PRESETS.items()}
    if break_pulse:
        presets["gemm"]["pulse_period"] = None
        presets["gemm"]["pulse_extra"] = 0
    if break_order:
        g, m = presets["gibbs"], presets["gemm"]
        gc = {k: g[k] for k in _CONTENT_KEYS if k in g}
        mc = {k: m[k] for k in _CONTENT_KEYS if k in m}
        for k in _CONTENT_KEYS:
            g.pop(k, None); m.pop(k, None)
        g.update(mc); m.update(gc)
    for ki, (kernel, archetype) in enumerate(schema.KERNELS):
        if level_only:
            p = dict(presets["gemm"]); p["pulse_period"] = None; p["pulse_extra"] = 0
            p["K0"] = LEVEL_ONLY_K0[archetype]
        elif one_preset:
            p = dict(ONE_PRESET)
        elif ord_case:
            cls = schema.ARCHETYPES[1:].index(archetype) % 2
            p = dict(K0=2048, content="double", churn=0.02, pulse_extra=2048)
            if ord_mode == "period":
                p["pulse_period"] = int(ord_periods[cls])
            else:
                p["pulse_period"] = int(ord_periods[1])
                p["pulse_burst"] = bool(cls)
        else:
            p = dict(presets[kernel])
        uniform_cases = level_only or one_preset or ord_case   # nothing but the declared difference may carry the label
        if cv_case == "random" and "k_noise" not in p and not uniform_cases:
            p["k_noise"] = random_noise[kernel]
        elif cv_case == "shot" and p.get("K0", 2048) > 0:
            p["k_noise"] = 2.0 / math.sqrt(p.get("K0", 2048))
        for r in range(int(reps)):
            specs.append(SynthSpec(name=kernel, seed=rep_seed(r, ki), n_pairs=int(n_pairs), rep_dir=1,
                                   label="synth", duration_s=duration_s, **p))
    for r in range(int(idle)):
        p = dict(IDLE_PRESET)
        name = p.pop("name")
        specs.append(SynthSpec(name=name, seed=rep_seed(r, len(schema.KERNELS)), n_pairs=int(n_pairs),
                               rep_dir=1, duration_s=duration_s, **p))
    return specs


def _corpus_worker(args: tuple) -> str:
    spec, root, compress, keep_sets = args
    return str(write_cell(spec, root, compress=compress, keep_sets=keep_sets, truth_sides=("t",)))


def write_corpus(root, *, compress: bool = True, keep_sets: bool = False, jobs: int = 1, **kw) -> dict:
    """Write the corpus of `corpus_specs(**kw)` under `root` and a `corpus.json` manifest
    (params, citation, every cell's path and spec). Each cell is seeded on its own, so
    `jobs > 1` (multiprocessing) writes byte-identical cells in less wall time. Corpus cells
    carry the "t"-side truth only (`per_seq_channels_t1` is null). Returns the manifest."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    specs = corpus_specs(**kw)
    tasks = [(s, str(root), compress, keep_sets) for s in specs]
    if int(jobs) > 1:
        import multiprocessing as mp
        # heaviest cells first, one task per dispatch, so no worker inherits all of one kernel's reps
        order = sorted(range(len(tasks)), key=lambda i: -(specs[i].K0 + specs[i].floor_F + specs[i].pulse_extra))
        paths = [None] * len(tasks)
        with mp.Pool(processes=int(jobs)) as pool:
            for i, path in zip(order, pool.imap(_corpus_worker, [tasks[i] for i in order], chunksize=1)):
                paths[i] = path
    else:
        paths = [_corpus_worker(t) for t in tasks]
    cells = [{"path": p, "spec": asdict(s)} for p, s in zip(paths, specs)]
    manifest = {
        "schema": "plan11.synth_corpus.v1",
        "params": {**{k: (list(v) if isinstance(v, tuple) else v) for k, v in kw.items()},
                   "compress": bool(compress), "keep_sets": bool(keep_sets), "jobs": int(jobs),
                   "presets": PRESETS, "idle_preset": IDLE_PRESET,
                   "floyd_pulse_extra_note": "al-Kindi SPEC review 2.4(d): pulse_extra = 2048 so k_jump finds boundaries"},
        "citation": CITATION_SYNTH,
        "n_cells": len(cells), "cells": cells,
        "generator_version": __version__,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(root / "corpus.json", "w") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


# ---------------------------------------------------------------------------
# CLI (SPEC 7.1)
# ---------------------------------------------------------------------------
def _int_list(s: str | None) -> tuple[int, ...]:
    if not s:
        return ()
    return tuple(int(x) for x in s.split(",") if x.strip())


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="synth.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cell", help="write one synthetic cell (SPEC 5.1)")
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
    c.add_argument("--counter-step-max", type=int, default=60)
    c.add_argument("--spin-bytes", default="2,8")
    c.add_argument("--double-words", type=int, default=64)
    c.add_argument("--decay-factor", type=float, default=0.7)
    c.add_argument("--floor-f", type=int, default=150)
    c.add_argument("--floor-churn", type=float, default=0.02)
    c.add_argument("--trend", type=float, default=0.0)
    c.add_argument("--seq-first", type=int, default=1)
    c.add_argument("--gap-seqs", default="")
    c.add_argument("--label", default="synth")
    c.add_argument("--no-compress", action="store_true")
    c.add_argument("--corrupt", choices=CORRUPTIONS, default=None)
    c.add_argument("--step", action="store_true", help="K0 for the first half, 3 K0 for the second (G1 must-fail)")
    c.add_argument("--k-decay", action="store_true", help="K also decays inside a pass (G-DEC (c) refusal)")
    c.add_argument("--k-decay-factor", type=float, default=0.5)
    c.add_argument("--l0-trend", type=float, default=0.0, help="linear drift of the l0 scale over the cell")
    c.add_argument("--rep-dir", type=int, default=1)
    c.add_argument("--row-order", choices=("ascending", "random"), default="ascending")
    c.add_argument("--pulse-burst", action="store_true", help="pulses in adjacent pairs every 2 * pulse_period")
    c.add_argument("--no-truth-sets", action="store_true")
    c.add_argument("--duration-s", type=float, default=None,
                   help="the cell's guest-running seconds recorded in truth.json (default n_pairs x 0.644; SPEC_epoch2 B12)")

    k = sub.add_parser("corpus", help="write the synthetic corpus (SPEC 5.2)")
    k.add_argument("--root", required=True)
    k.add_argument("--n-pairs", type=int, default=120)
    k.add_argument("--reps", type=int, default=8)
    k.add_argument("--idle", type=int, default=8)
    k.add_argument("--seed", type=int, default=20260916)
    k.add_argument("--break-pulse", action="store_true")
    k.add_argument("--break-order", action="store_true")
    k.add_argument("--cv-case", choices=("random", "shot", "preset"), default="random")
    k.add_argument("--level-only", action="store_true")
    k.add_argument("--one-preset", action="store_true")
    k.add_argument("--ord-case", action="store_true")
    k.add_argument("--ord-periods", default="6,12")
    k.add_argument("--ord-mode", choices=("period", "burst"), default="period")
    k.add_argument("--no-compress", action="store_true")
    k.add_argument("--truth-sets", action="store_true", help="keep the sorted page lists in truth.json")
    k.add_argument("--jobs", type=int, default=1, help="processes writing cells in parallel (results identical)")
    k.add_argument("--duration-s", type=float, default=None,
                   help="every cell's guest-running seconds recorded in truth.json (default n_pairs x 0.644; SPEC_epoch2 B12)")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = _build_parser()
    a = ap.parse_args(argv)
    try:
        if a.cmd == "cell":
            sb = _int_list(a.spin_bytes)
            if len(sb) != 2:
                ap.error("--spin-bytes needs two integers, e.g. 2,8")
            spec = SynthSpec(
                name=a.name, seed=a.seed, n_pairs=a.n_pairs, K0=a.k0, k_noise=a.k_noise, churn=a.churn,
                pulse_period=a.pulse_period, pulse_extra=a.pulse_extra, content=a.content,
                counter_step_max=a.counter_step_max, spin_bytes=(sb[0], sb[1]), double_words=a.double_words,
                decay_factor=a.decay_factor, floor_F=a.floor_f, floor_churn=a.floor_churn, trend=a.trend,
                seq_first=a.seq_first, gap_seqs=_int_list(a.gap_seqs), label=a.label,
                step_factor=3.0 if a.step else 1.0,
                k_decay_factor=(a.k_decay_factor if a.k_decay else None),
                l0_trend=a.l0_trend, corrupt=a.corrupt, rep_dir=a.rep_dir, row_order=a.row_order,
                pulse_burst=a.pulse_burst, duration_s=a.duration_s,
            )
            cell = write_cell(spec, a.root, compress=not a.no_compress,
                              keep_sets=(False if a.no_truth_sets else None))
            print(f"[synth] wrote {cell}")
            return 0
        if a.cmd == "corpus":
            per = _int_list(a.ord_periods)
            if len(per) != 2:
                ap.error("--ord-periods needs two integers, e.g. 6,12")
            m = write_corpus(a.root, compress=not a.no_compress, keep_sets=a.truth_sets, jobs=a.jobs,
                             n_pairs=a.n_pairs, reps=a.reps, idle=a.idle, seed=a.seed,
                             break_pulse=a.break_pulse, break_order=a.break_order, cv_case=a.cv_case,
                             level_only=a.level_only, one_preset=a.one_preset, ord_case=a.ord_case,
                             ord_periods=(per[0], per[1]), ord_mode=a.ord_mode, duration_s=a.duration_s)
            print(f"[synth] wrote {m['n_cells']} cells under {a.root}")
            return 0
        ap.error(f"unknown command {a.cmd}")
        return 1
    except FileNotFoundError as exc:
        print(f"missing input: {exc}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
