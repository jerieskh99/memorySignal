"""Shared set-up for the detection-layer tests (SPEC_DETECTION.md 4.5): sys.path, the synthetic
two-class corpora built once per session, the chain up to G-K0 two-class, and small helpers.

Three corpora, each built at most once per pytest session (or reused from ``$PLAN11_DET_CACHE``
when that directory holds a finished build, for iterating on the tests):

  main    the reduced corpus of SPEC_DETECTION 4.5 (six kernels x 3 reps, 4 idle, members 1 to 8 at
          120 pairs; member 8 at 10,000 pages instead of 40,000 for cost; classes.csv with order_index;
          iteration_boundaries.csv), order_confound off, one campaign label;
  on      the confounded corpus: six kernels spanning three archetypes, members 1 to 8, eight idle cells,
          order_confound on, campaign_labels confounded, idle_campaigns 2, drift_level at rate 0.02 (the
          audible cases);
  tiny    the stage-2 fixture: two kernels, members 1 and 2 at a different pair count (the cadence
          leak), the re-launched and harness-idle cells; its classes.csv without order_index.

No server path, no real data, no sandbox workload name.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG.parent))
sys.path.insert(0, str(HERE))

from plan11_encoding_ladder import series as S  # noqa: E402
from plan11_encoding_ladder import verdicts as V  # noqa: E402
from plan11_encoding_ladder import classes as C  # noqa: E402
from plan11_encoding_ladder import synth_detection as SD  # noqa: E402
from plan11_encoding_ladder import gates_precondition as GP  # noqa: E402
from plan11_encoding_ladder import gates_detection as GD  # noqa: E402
from plan11_encoding_ladder import detection_metrics as DM  # noqa: E402
from plan11_encoding_ladder import detection_splits as DS  # noqa: E402
from plan11_encoding_ladder import detection_levels as DL  # noqa: E402
from plan11_encoding_ladder import extract as EX  # noqa: E402

N_EST = 20          # a cheap forest for the tests; the SPEC's 300 is the default of every module
RUNGS = ("apf", "wapf", "persist", "content", "combined")
MAIN_KERNELS = "gemm,floyd,gibbs,histogram,fft,lexer"
ON_KERNELS = "gemm,floyd,fft,histogram,lexer,rmat_gen"
CORPORA = {
    "main": dict(n_pairs=120, reps=3, idle=4, kernels=MAIN_KERNELS, members="all", member8_k0=10000, write_classes=True, write_boundaries=True),
    "on": dict(n_pairs=60, reps=3, idle=8, kernels=ON_KERNELS, members="all", member8_k0=10000, order_confound="on", campaign_labels="confounded",
               idle_campaigns=2, drift_level=True, drift_rate=0.02, write_classes=True),
    "tiny": dict(n_pairs=40, reps=2, idle=2, kernels="gemm,gibbs", members="1,2", sandbox_n_pairs=32, stage2_fixture=True, write_classes=True),
}
_BUILT: dict = {}
_SESSION_TMP: Path | None = None


def session_tmp() -> Path:
    global _SESSION_TMP
    if _SESSION_TMP is None:
        cache = os.environ.get("PLAN11_DET_CACHE")
        _SESSION_TMP = Path(cache) if cache else Path(tempfile.mkdtemp(prefix="plan11_det_"))
        _SESSION_TMP.mkdir(parents=True, exist_ok=True)
    return _SESSION_TMP


def tmp_out() -> Path:
    return Path(tempfile.mkdtemp(prefix="plan11_det_t_"))


def rows(path: Path) -> list[dict]:
    return S.read_csv(path)


def row_where(path: Path, **match) -> dict:
    for r in rows(path):
        if all(str(r.get(k)) == str(v) for k, v in match.items()):
            return r
    raise AssertionError(f"no row matching {match} in {path}")


def jload(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def run_cli(module: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(PKG.parent)}
    return subprocess.run([sys.executable, "-m", f"plan11_encoding_ladder.{module}", *[str(a) for a in args]], cwd=str(PKG.parent), env=env,
                          capture_output=True, text=True, check=check)


def _chain(root: Path, out: Path, kw: dict, classes_name: str = "classes.csv") -> Path:
    """extract index, classes apply, extract all, preconditions (c1 0.001), inherit-selection --default W8_H4,
    series features for the five rungs, gk0, admissibility, gk0-cells, gn (SPEC_DETECTION 4.5)."""
    EX.build_index(root, out / "cells.csv")
    res = C.apply_classes(out, root / classes_name)
    assert res["status"] == "ok", res
    EX.extract_all(out / "cells.csv", out, jobs=4)
    GP.gate_preconditions(out, assume_failed_zero=True, assume_reason="synthetic", c1_activity_min=0.001)
    C.inherit_selection(out, default_grid="W8_H4")
    if (root / "iteration_boundaries.csv").is_file():
        shutil.copyfile(root / "iteration_boundaries.csv", out / "inputs" / "iteration_boundaries.csv")
    for rung in RUNGS:
        for norm in ((True,) if rung == "combined" else (False, True)):
            S.build_features(out, None, rung, 8, 4, norm, {})
    GP.gate_gk0(out)
    GD.admissibility(out)
    GD.gk0_sandbox_template(out)
    GD.gk0_cells(out)
    GD.gn_two_class(out)
    return out


def corpus(tag: str = "main") -> Path:
    """The out dir of one corpus after the chain; built once per session."""
    if tag in _BUILT:
        return _BUILT[tag]
    base = session_tmp() / tag
    root, out = base / "root", base / "out"
    marker = out / "gates" / "detection" / "gn.csv"
    if not marker.is_file():
        if base.exists():
            shutil.rmtree(base)
        kw = dict(CORPORA[tag])
        wc = kw.pop("write_classes", False); wb = kw.pop("write_boundaries", False)
        SD.write_corpus(root, compress=True, jobs=4, write_classes=wc, write_boundaries=wb, **kw)
        out.mkdir(parents=True, exist_ok=True)
        _chain(root, out, kw, classes_name="classes.no_order.csv" if tag == "tiny" else "classes.csv")
    _BUILT[tag] = out
    return out


def corpus_root(tag: str = "main") -> Path:
    corpus(tag)
    return session_tmp() / tag / "root"


_SPLITS: dict = {}


def split(tag: str, rung: str, name: str = "lowo", **kw) -> Path:
    """A split run once per session and reused: lowo/loco/lofo on the norm features with a cheap forest."""
    key = (tag, rung, name, tuple(sorted(kw.items())))
    if key in _SPLITS:
        return _SPLITS[key]
    out = corpus(tag)
    args = dict(normalized=True, n_perm=6, n_perm_required=6, n_estimators=N_EST, run_null=(name in ("lowo", "loco")))
    args.update(kw)
    d = DM.run_detection_split(out, rung, "W8_H4", name, **args)
    _SPLITS[key] = d
    return d


def scores(d: Path) -> dict:
    return jload(Path(d) / "scores.json")


def write_fixture_scores(out: Path, rung: str, name: str, payload: dict, normalized: bool = True) -> Path:
    """A hand-written scores.json (the schema of SPEC_DETECTION 3.3.7) for the gate tests."""
    d = DM.split_dir(out, rung, "W8_H4", name, normalized)
    d.mkdir(parents=True, exist_ok=True)
    doc = {"status": "ok", "split": name, "tpr_05": 0.5, "fpr_05_realized": 0.05, "tpr_01": 0.25, "auc": 0.9, "null": {"status": "ok", "tpr05": {"verdict": V.PASS, "p95": 0.1, "rank_text": "rank 500 of 500"},
                                                                                                                       "auc": {"verdict": V.PASS, "p95": 0.6}}}
    doc.update(payload)
    S.write_json(d / "scores.json", "plan11.detection.scores.v1", {"rung": rung, "grid_id": "W8_H4"}, "fixture", doc)
    return d


def fixture_selection(out: Path) -> None:
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    if not (out / "cells.csv").is_file():
        S.write_csv(out / "cells.csv", ("cell_id",), [])
    C.inherit_selection(out, default_grid="W8_H4")


class restrict_admissible:
    """Temporarily mark the cells selected by ``exclude`` (a predicate on the admissibility row) as
    inadmissible in gates/detection/admissibility.csv; restores the file on exit."""

    def __init__(self, out: Path, exclude):
        self.p = C.det_dir(out) / "admissibility.csv"; self.exclude = exclude; self.bak = None

    def __enter__(self):
        self.bak = self.p.read_bytes()
        rs = S.read_csv(self.p)
        for r in rs:
            if self.exclude(r):
                r["admissible"] = "false"; r["admissible_pair_rungs"] = "false"; r["reason"] = "test: excluded"
        S.write_csv(self.p, GD.ADM_COLUMNS, rs)
        return self.p

    def __exit__(self, *a):
        self.p.write_bytes(self.bak)
        return False


class swap_file:
    """Temporarily replace (or remove, with content None) a file; restores it on exit."""

    def __init__(self, path: Path, content: bytes | str | None):
        self.path = Path(path); self.content = content; self.bak = None

    def __enter__(self):
        self.bak = self.path.read_bytes() if self.path.is_file() else None
        if self.content is None:
            if self.path.is_file():
                self.path.unlink()
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_bytes(self.content if isinstance(self.content, bytes) else self.content.encode())
        return self.path

    def __exit__(self, *a):
        if self.bak is None:
            if self.path.is_file():
                self.path.unlink()
        else:
            self.path.write_bytes(self.bak)
        return False
