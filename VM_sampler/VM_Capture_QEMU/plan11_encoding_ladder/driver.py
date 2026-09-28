#!/usr/bin/env python3
"""driver.py -- SPEC section 1's name for the driver; the implementation is `run_moves.py`
(the build brief's name). Both invoke the same `main`."""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from plan11_encoding_ladder.run_moves import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
