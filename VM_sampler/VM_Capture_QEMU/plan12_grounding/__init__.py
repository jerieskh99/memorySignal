"""plan12_grounding: the engine of the grounding paper (working name; `memorySignal/grounding_paper/`).

The contract is `plan12_grounding/SPEC.md`. Slice 1 (2026-09-30): the driver and its record book
(`run_moves.py`), move 0 (`inputs.py`), move 1 (`extract.py`), move 2 (`sanity.py`) and the smoke
corpus writer (`synth_grounding.py`). Every module runs from `VM_sampler/VM_Capture_QEMU/` as
`python3 -m plan12_grounding.<module>`; the encoding toolkit (`plan11_encoding_ladder`) and the
console (`plan10_analysis`) are imported, never written to (SPEC 1.5).

No sandbox workload is named anywhere in this package (SPEC 1.1); recordings outside the twelve
kernels and idle are counted, never named.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

__version__ = "0.1.0"

HERE = Path(__file__).resolve().parent
PACKAGE_NAME = "plan12_grounding"


def toolkit_fingerprint() -> dict:
    """The code that ran (SPEC 1.4, 7): the sha256 of every `plan12_grounding/*.py`, and one sha256
    over their sorted (name, sha256) pairs. Recorded in the record book and in every series file."""
    files = {}
    for p in sorted(HERE.glob("*.py")):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        files[p.name] = h.hexdigest()
    top = hashlib.sha256("\n".join(f"{k} {v}" for k, v in sorted(files.items())).encode("utf-8")).hexdigest()
    return {"sha256": top, "n_py_files": len(files), "files": files, "package_version": __version__}
