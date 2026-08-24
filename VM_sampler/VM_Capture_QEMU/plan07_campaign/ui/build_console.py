#!/usr/bin/env python3
"""Build capture_console.html by injecting the pipeline's real data into the template.

The UI (capture_console.template.html) hardcodes NOTHING about the workloads,
scaling rules, or metric taxonomy. This script reads those from the single
source of truth -- the Python pipeline itself -- and injects them, so the UI can
never silently drift when the Python changes.

Sources of truth:
  full_campaign_steps.txt        -> BASE_CMDS (workload -> command)
  generate_database_steps.py     -> SCALABLE, CLAMP_MB
  subset_run.py                  -> FEATURES_BY_GROUP, SUBMODULE_COLUMNS

Run after any change to those files:
    python3 ui/build_console.py
-> rewrites ui/capture_console.html

CAPPED (workloads pinned at scale 1.0) is DERIVED here, not hardcoded: a workload
is pinned when its base value for a CLAMP_MB flag already meets/exceeds the cap,
so max(cap, base) == base and any scale > 1.0 clamps (matches the max(cap,base)
ceiling in generate_database_steps.scale_command).
"""
from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
sys.path.insert(0, str(CAMPAIGN))

from generate_database_steps import SCALABLE, CLAMP_MB, binary_of   # noqa: E402
from subset_run import FEATURES_BY_GROUP, SUBMODULE_COLUMNS         # noqa: E402

BASE_STEPS = CAMPAIGN / "full_campaign_steps.txt"
TEMPLATE = HERE / "capture_console.template.html"
OUT = HERE / "capture_console.html"
MARKER = "/*@@GENERATED_DATA@@*/"


def base_cmds() -> dict[str, str]:
    cmds = [l.strip() for l in BASE_STEPS.read_text().splitlines()
            if l.strip() and not l.strip().startswith("#")]
    return {binary_of(c): c for c in cmds}


def derive_capped(cmds: dict[str, str]) -> dict[str, float]:
    """Workloads whose base value meets/exceeds a CLAMP_MB cap -> max safe scale.

    Only emits the pinned ones (max_safe_scale <= 1.0), which is what the UI
    marks with the pin glyph. Mirrors ceiling = max(cap, base) from scale_command.
    """
    # A flag only clamps if it is BOTH scaled and capped -- a capped-but-not-scalable
    # flag (e.g. --mem-cap-mb) is never multiplied, so its cap never fires.
    clamps = set(CLAMP_MB) & set(SCALABLE)
    capped: dict[str, float] = {}
    for wk, cmd in cmds.items():
        toks = shlex.split(cmd)
        worst = None
        for i, t in enumerate(toks):
            if t in clamps and i + 1 < len(toks) and toks[i + 1].isdigit():
                base = int(toks[i + 1])
                cap = CLAMP_MB[t]
                ceiling = max(cap, base)
                safe = ceiling / base           # max scale before this flag clamps
                worst = safe if worst is None else min(worst, safe)
        if worst is not None and worst <= 1.0:
            capped[wk] = round(worst, 4)
    return capped


def js_const(name: str, value, set_wrap: bool = False) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":") if not isinstance(value, dict) or len(value) > 12 else (", ", ": "))
    if set_wrap:
        return f"const {name}=new Set({json.dumps(sorted(value))});"
    return f"const {name}={payload};"


def build() -> int:
    cmds = base_cmds()
    capped = derive_capped(cmds)

    blocks = [
        "const BASE_CMDS = " + json.dumps(cmds, ensure_ascii=False, indent=2) + ";",
        js_const("SCALABLE", sorted(SCALABLE), set_wrap=True),
        "const CLAMP = " + json.dumps(CLAMP_MB, ensure_ascii=False) + ";",
        "const FEATURES_BY_GROUP = " + json.dumps(FEATURES_BY_GROUP, ensure_ascii=False) + ";",
        "const SUBMODULE_COLUMNS = " + json.dumps(SUBMODULE_COLUMNS, ensure_ascii=False) + ";",
        "const CAPPED = " + json.dumps(capped, ensure_ascii=False) + ";",
    ]
    data = "\n".join(blocks)

    template = TEMPLATE.read_text()
    if MARKER not in template:
        sys.exit(f"marker {MARKER} not found in {TEMPLATE.name}")
    html = template.replace(MARKER, data)
    OUT.write_text(html)

    print(f"[build_console] wrote {OUT.name}")
    print(f"  BASE_CMDS: {len(cmds)} workloads")
    print(f"  SCALABLE: {len(SCALABLE)} flags   CLAMP: {len(CLAMP_MB)} flags")
    print(f"  FEATURES_BY_GROUP: {len(FEATURES_BY_GROUP)} groups   "
          f"SUBMODULE_COLUMNS: {len(SUBMODULE_COLUMNS)} submodules")
    print(f"  CAPPED (pinned at 1.0, derived): {list(capped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
