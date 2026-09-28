"""Shared test set-up for builder 2's tests: sys.path, the in-test generator, tiny helpers."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG.parent))      # VM_Capture_QEMU: makes plan11_encoding_ladder importable
sys.path.insert(0, str(HERE))

import _synth_b2 as SY  # noqa: E402
from plan11_encoding_ladder import series as S  # noqa: E402
from plan11_encoding_ladder import verdicts as V  # noqa: E402

SMALL_KERNELS = ("gemm", "floyd", "gibbs", "histogram", "fft", "lexer")


def tmp_out() -> Path:
    return Path(tempfile.mkdtemp(prefix="plan11_b2_"))


def corpus(out: Path | None = None, **kw) -> Path:
    """Write a corpus of SPEC 5.2 (with variants) into a fresh out dir; returns the out dir."""
    out = out or tmp_out()
    SY.write_corpus(out, SY.corpus_specs(**kw))
    return out


def rows(path: Path) -> list[dict]:
    return S.read_csv(path)


def row_where(path: Path, **match) -> dict:
    for r in rows(path):
        if all(str(r.get(k)) == str(v) for k, v in match.items()):
            return r
    raise AssertionError(f"no row matching {match} in {path}")


def jload(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def pass_table_with(out: Path, entries: dict) -> None:
    """Write inputs/pass_table.csv with {kernel: (passes, source)}."""
    from plan11_encoding_ladder import gates_calibration as GC
    GC.write_pass_table_template(out / "inputs" / "pass_table.csv")
    rs = S.read_csv(out / "inputs" / "pass_table.csv")
    for r in rs:
        if r["kernel"] in entries:
            passes, src = entries[r["kernel"]]
            r["passes_per_600s"] = "" if passes is None else passes
            r["source"] = src
    S.write_csv(out / "inputs" / "pass_table.csv", GC.PASS_TABLE_COLUMNS, rs)


def admissibility(out: Path) -> None:
    from plan11_encoding_ladder import gates_precondition as GP
    (out / "inputs").mkdir(exist_ok=True)
    (out / "inputs" / "idle_admissibility.json").write_text(json.dumps({k: "synthetic" for k in GP.ADMISSIBILITY_KEYS}))
