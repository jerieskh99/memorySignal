#!/usr/bin/env python3
"""Smoke + correctness test for plan08_b1/b1_extract_hamming.py.

The load-bearing check: the APF the extractor derives from the substrate CSV
equals plan02_apf_helper._compute_active_page_fraction on the SAME page pair,
exactly. Both define APF as "fraction of pages with any byte changed", so if
they ever disagree the extractor is wrong.

Run:  python3 tests/test_plan08_b1_extract.py      (plain asserts, no pytest dep)
      pytest tests/test_plan08_b1_extract.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

QEMU_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QEMU_DIR))
import plan02_apf_helper as helper                                   # noqa: E402

EXTRACT = QEMU_DIR / "plan08_b1" / "b1_extract_hamming.py"
PAGE = 4096
BITS_P = PAGE * 8


def _popcount_bytes(a: np.ndarray, b: np.ndarray) -> int:
    """Bits flipped between two equal-length byte arrays -- the differ's hamming."""
    return int(np.unpackbits(np.bitwise_xor(a, b)).sum())


def _write_raw(path: Path, pages: np.ndarray) -> None:
    path.write_bytes(pages.tobytes())


def _run(csv_path: Path, out_dir: Path, n_pages: int, extra=()) -> None:
    cmd = [sys.executable, str(EXTRACT), str(csv_path), str(out_dir),
           "--n-pages", str(n_pages), "--page-size", str(PAGE), *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"extractor failed:\n{r.stdout}\n{r.stderr}"


def _read_traj(out_dir: Path) -> tuple[list[dict], dict]:
    lines = [json.loads(l) for l in (out_dir / "b1_trajectory.jsonl").read_text().splitlines() if l]
    body = [d for d in lines if not d.get("final")]
    sentinel = next(d for d in lines if d.get("final"))
    return body, sentinel


def build_case(tmp: Path):
    """10 pages. seq0: pages {2,5,7} change (light, full-rewrite, 1-bit).
       seq1: pages {0,2} change. Everything else identical."""
    rng = np.random.default_rng(1234)
    n_pages = 10
    base = rng.integers(0, 256, size=(n_pages, PAGE), dtype=np.uint8)

    s0 = base.copy()
    s1 = base.copy()                       # "curr" of pair 0 / "prev" of pair 1
    s2 = s1.copy()

    # pair seq0: base -> s1
    s1[2, :8] ^= 0x01                      # light: 8 bits
    s1[5] = rng.integers(0, 256, size=PAGE, dtype=np.uint8)   # full rewrite
    s1[7, 0] ^= 0x01                       # 1 bit
    # pair seq1: s1 -> s2
    s2[0, :4] ^= 0xFF                      # 32 bits
    s2[2] = s1[2]                          # (unchanged this pair)
    s2[2, 100] ^= 0x0F                     # 4 bits on page 2 again

    dumps = tmp / "dumps"; dumps.mkdir()
    p0, p1, p2 = dumps / "s0.raw", dumps / "s1.raw", dumps / "s2.raw"
    _write_raw(p0, s0); _write_raw(p1, s1); _write_raw(p2, s2)

    # reference APF from the REAL helper, per pair
    ref_apf = {
        0: helper._compute_active_page_fraction(p0, p1, PAGE),
        1: helper._compute_active_page_fraction(p1, p2, PAGE),
    }

    # build the substrate-style CSV the consumer would have written: one row per
    # changed page per seq, cols seq,page_index,hamming,cosine (cosine arbitrary).
    rows = []
    ham_expect = {0: 0, 1: 0}
    for seq, (prev, curr) in {0: (s0, s1), 1: (s1, s2)}.items():
        for pg in range(n_pages):
            h = _popcount_bytes(prev[pg], curr[pg])
            if h == 0:
                continue
            cos = 0.5                       # placeholder distance; extractor stores raw
            rows.append((seq, pg, h, cos))
            ham_expect[seq] += h
    csv_path = tmp / "substrate_trajectory.csv"
    with csv_path.open("w") as fh:
        fh.write("seq,page_index,hamming,cosine\n")
        for seq, pg, h, cos in rows:
            fh.write(f"{seq},{pg},{h},{cos}\n")

    changed = {0: sum(1 for r in rows if r[0] == 0),
               1: sum(1 for r in rows if r[0] == 1)}
    return csv_path, n_pages, ref_apf, ham_expect, changed


def test_extract_matches_helper_and_field(tmp_path=None):
    tmp = Path(tmp_path or tempfile.mkdtemp())
    csv_path, n_pages, ref_apf, ham_expect, changed = build_case(tmp)

    # --- default run: field written (zstd) ---
    out = tmp / "out"
    _run(csv_path, out, n_pages)

    body, sentinel = _read_traj(out)
    assert sentinel["n_pairs"] == 2, sentinel
    by_seq = {d["seq"]: d for d in body}

    for seq in (0, 1):
        d = by_seq[seq]
        assert d["n_changed"] == changed[seq], (seq, d, changed)
        assert d["ham_sum"] == ham_expect[seq], (seq, d, ham_expect)
        derived_apf = d["n_changed"] / d["n_pages"]
        # THE load-bearing equality: extractor's APF == the real helper's APF
        assert abs(derived_apf - ref_apf[seq]) < 1e-12, (seq, derived_apf, ref_apf[seq])
        derived_wapf = d["ham_sum"] / (d["n_pages"] * BITS_P)
        assert derived_wapf <= derived_apf + 1e-12, (seq, derived_wapf, derived_apf)  # wAPF <= APF

    # --- field content: decompress and check it round-trips the changed rows ---
    field_zst = out / "hc_field.csv.zst"
    assert field_zst.is_file(), "compressed field not written by default"
    text = subprocess.run(["zstd", "-dc", "-q", str(field_zst)],
                          capture_output=True, text=True, check=True).stdout
    flines = [l for l in text.splitlines() if l]
    assert flines[0] == "seq,page_index,hamming,cosine"
    assert len(flines) - 1 == sum(changed.values()), (len(flines) - 1, changed)
    assert ",0.5" in flines[1], "cosine not stored raw"

    # --- --no-field: no field file, trajectory identical ---
    out2 = tmp / "out_nofield"
    _run(csv_path, out2, n_pages, extra=["--no-field"])
    assert not (out2 / "hc_field.csv.zst").exists()
    assert not (out2 / "hc_field.csv").exists()
    body2, _ = _read_traj(out2)
    assert body2 == body, "trajectory differs with/without field"

    # --- .zst INPUT reads the same ---
    csv_zst = tmp / "substrate_trajectory.csv.zst"
    subprocess.run(["zstd", "-q", "-f", "-o", str(csv_zst), str(csv_path)], check=True)
    out3 = tmp / "out_zstin"
    _run(csv_zst, out3, n_pages)
    body3, _ = _read_traj(out3)
    assert body3 == body, "zst-input trajectory differs from plain"

    print("PASS: extractor APF == plan02_apf_helper; ham_sum, field, --no-field, .zst-input all OK")


if __name__ == "__main__":
    test_extract_matches_helper_and_field()
