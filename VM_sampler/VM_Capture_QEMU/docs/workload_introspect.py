#!/usr/bin/env python3
"""Extract, from a workload's own source, its PARAMETERS and its EDGE CASES.

Both are read straight from the source so they cannot drift:

  parameters -- every tunable flag with its default. C workloads read flags via
    p2_get_i64/u64/f64/str(argc, argv, "--flag", DEFAULT); Python workloads use
    argparse add_argument(...). Boolean flags (store_true / bare string checks)
    are captured too.

  edges -- the workload's own guard rails: the P2_LOG_ERR / validation messages
    that fire on a bad parameter or a failed allocation (these state the valid
    ranges and the failure modes), plus any CAVEAT / HONEST / Safety / LIMITATION
    note the author wrote in the header.

Used by make_workload_algorithms.py to render a "Parameters" tab (default vs the
value chosen in the campaign) and an "Edge cases" tab per workload.
"""
from __future__ import annotations

import re
import shlex
from pathlib import Path

# base_argparser (common/phase2_common.py) flags shared by every Python workload
PY_BASE_PARAMS = [
    ("--duration", "None (run once)"),
    ("--seed", "42"),
    ("--output-dir", "None"),
    ("--sandbox-dir", "None"),
    ("--safe-root", "None"),
    ("--phase-markers", "off (store_true)"),
    ("--dry-run", "off (store_true)"),
    ("--cleanup", "off (store_true)"),
    ("--cpu-affinity", "None"),
    ("--verbose", "off (store_true)"),
]

_C_GET = re.compile(
    r'p2_get_(?:i64|u64|f64|str|bool)\s*\(\s*argc\s*,\s*argv\s*,\s*"(--[A-Za-z0-9-]+)"\s*,\s*([^)]*?)\)'
)
_C_BOOLFLAG = re.compile(r'p2_(?:has_flag|get_flag)\s*\(\s*argc\s*,\s*argv\s*,\s*"(--[A-Za-z0-9-]+)"')
_C_BARE_BOOL = re.compile(r'"(--(?:phase-markers|no-mlock|dry-run|cleanup|verbose))"')
_PY_ADD = re.compile(
    r'add_argument\(\s*"(--[A-Za-z0-9-]+)"(.*?)\)', re.S)


def extract_params(src: Path):
    """Return ordered list of (flag, default_str) for a workload source."""
    txt = src.read_text(errors="ignore")
    params = []
    seen = set()

    def add(flag, default):
        if flag not in seen:
            seen.add(flag)
            params.append((flag, default.strip()))

    if src.suffix == ".c":
        for m in _C_GET.finditer(txt):
            add(m.group(1), m.group(2))
        for m in _C_BOOLFLAG.finditer(txt):
            add(m.group(1), "off (flag)")
        for m in _C_BARE_BOOL.finditer(txt):
            add(m.group(1), "off (flag)")
    else:
        # per-workload argparse first (most specific), then the shared base set
        for m in _PY_ADD.finditer(txt):
            flag, rest = m.group(1), m.group(2)
            if "store_true" in rest:
                default = "off (store_true)"
            else:
                dm = re.search(r'default\s*=\s*([^,)]+)', rest)
                default = dm.group(1).strip() if dm else "(required)"
            add(flag, default)
        for flag, default in PY_BASE_PARAMS:
            add(flag, default)
    return params


def _clean_msg(s: str) -> str:
    """Tidy a C printf-style error string into a readable edge note."""
    s = s.replace("\\n", " ").strip()
    # drop format specifiers but keep the human words + numeric ranges
    s = re.sub(r"%[0-9.]*(?:ll|l|z)?[dinuxsfg]", "", s)
    s = re.sub(r"\s{2,}", " ", s).strip(" :")
    return s


def extract_edges(src: Path):
    """Return a list of edge-case / limit strings taken from the source."""
    txt = src.read_text(errors="ignore")
    edges = []
    seen = set()

    def add(kind, text):
        text = text.strip()
        key = (kind, text.lower())
        if text and key not in seen and len(text) > 3:
            seen.add(key)
            edges.append((kind, text))

    if src.suffix == ".c":
        # validation + failure messages the workload prints on a bad input/alloc
        for m in re.finditer(r'P2_LOG_ERR\s*\(\s*"([^"]+)"', txt):
            msg = _clean_msg(m.group(1))
            low = msg.lower()
            if any(k in low for k in ("range", "exceed", "too small", "too large",
                                      "invalid", "must", "fail", "cannot", "denied",
                                      "at least", "positive", ">=", "<=")):
                add("guard", msg)
        # header caveat / safety / limitation notes: capture the labelled line
        # and its continuation, stopping at a blank comment line or a new header
        # (a "Word:" label or a "----" rule) so we don't swallow the next section.
        hm = re.match(r"\s*/\*(.*?)\*/", txt, re.S)
        if hm:
            lines = [re.sub(r"^\s?\*\s?", "", ln).rstrip()
                     for ln in hm.group(1).splitlines()]
            i = 0
            label = re.compile(r'^\s*(HONEST CAVEAT|CAVEAT|Safety note|Safety notes|'
                               r'LIMITATION|LIMITATIONS|EDGE CASE|EDGE CASES)\b', re.I)
            stop = re.compile(r'^\s*([A-Z][A-Za-z ]{1,24}:|-{3,}|=+\s*$)')
            while i < len(lines):
                if label.search(lines[i]):
                    buf = [re.sub(r'^\s*(HONEST CAVEAT|CAVEAT|Safety notes?|'
                                  r'LIMITATIONS?|EDGE CASES?)\b[:.\s-]*', '',
                                  lines[i], flags=re.I).strip()]
                    j = i + 1
                    while j < len(lines):
                        nxt = lines[j]
                        if not nxt.strip() or stop.match(nxt):
                            break
                        buf.append(nxt.strip())
                        j += 1
                    text = re.sub(r"\s{2,}", " ", " ".join(buf)).strip()
                    if text:
                        add("caveat", text[:500])
                    i = j
                else:
                    i += 1
    else:
        for m in re.finditer(r'(?:raise\s+\w+|P2_LOG_ERR|log_err|sys\.exit)\([^)]*?["\']([^"\']{6,})["\']', txt):
            add("guard", _clean_msg(m.group(1)))
        for m in re.finditer(r'choices\s*=\s*(range\([^)]*\)|\[[^\]]*\]|\([^)]*\))', txt):
            add("guard", f"allowed values: {m.group(1).strip()}")
    return edges


def parse_invocation(cmd: str):
    """Parse a campaign command into {flag: value}. Bare flags map to True."""
    try:
        toks = shlex.split(cmd)
    except ValueError:
        toks = cmd.split()
    chosen = {}
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                chosen[t] = toks[i + 1]
                i += 2
            else:
                chosen[t] = True
                i += 1
        else:
            i += 1
    return chosen


if __name__ == "__main__":
    import sys
    for f in sys.argv[1:]:
        p = Path(f)
        print(f"== {p.name} ==")
        print(" params:")
        for fl, d in extract_params(p):
            print(f"   {fl:20s} default={d}")
        print(" edges:")
        for kind, e in extract_edges(p):
            print(f"   [{kind}] {e}")
