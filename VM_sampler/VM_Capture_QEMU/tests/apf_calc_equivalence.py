#!/usr/bin/env python3
"""apf_calc <-> Python-helper equivalence check (Plan 05 Wave 4, Step 5).

Proves the Rust port is correct: the same synthetic dump pair fed to the Rust
`apf_calc` binary and to `plan02_apf_helper._compute_active_page_fraction` must
return the IDENTICAL APF. Deterministic integer page-counting, so the match is
exact (not just close).

Needs the built binary, so run on the server after `cargo build --release`:

    python3 tests/apf_calc_equivalence.py \
        ../VM_Capture/apf_calc/target/release/apf_calc

Exit 0 = equivalent; non-zero = mismatch (prints both values).
"""
from __future__ import annotations

import glob
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plan02_apf_helper import _compute_active_page_fraction  # noqa: E402

PAGE = 4096


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: apf_calc_equivalence.py <path-to-apf_calc-binary>")
        return 2
    binary = sys.argv[1]
    if not Path(binary).is_file():
        print(f"binary not found: {binary}")
        return 2

    with tempfile.TemporaryDirectory() as d:
        dd = Path(d)
        a, b, out = dd / "a.raw", dd / "b.raw", dd / "out"
        out.mkdir()

        # 10 identical pages; flip one byte in pages 0, 4, 9 of b -> 3/10 differ.
        pa = bytearray(PAGE * 10)
        pb = bytearray(pa)
        pb[0 * PAGE + 0] = 1
        pb[4 * PAGE + 100] = 9
        pb[9 * PAGE + (PAGE - 1)] = 255
        a.write_bytes(pa)
        b.write_bytes(pb)

        # Rust apf_calc -> reads the single value it wrote
        subprocess.run([binary, str(a), str(b), str(out)], check=True)
        files = sorted(glob.glob(str(out / "apf" / "apf_results_par-*.txt")))
        if not files:
            print("apf_calc wrote no output file")
            return 1
        rust_apf = float(Path(files[-1]).read_text().strip())

        # Python helper (the reference math)
        py_apf = _compute_active_page_fraction(a, b, PAGE)

        print(f"rust_apf = {rust_apf}")
        print(f"py_apf   = {py_apf}")
        print(f"expected = 0.3")

        ok = (
            abs(rust_apf - 0.3) < 1e-12
            and abs(py_apf - 0.3) < 1e-12
            and abs(rust_apf - py_apf) < 1e-12
        )
        if ok:
            print("EQUIVALENCE OK: apf_calc == plan02_apf_helper == 0.3")
            return 0
        print("EQUIVALENCE FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
