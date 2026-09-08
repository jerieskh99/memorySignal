#!/usr/bin/env python3
"""channel_roster.py -- the 64 metric channels and the speed level each one dies at.

Derived, not typed. Two sources, both the differ's own:

  live_delta_calc_modular/src/metrics/mod.rs          csv_header()  -> the 64 names, in order
  live_delta_calc_modular/src/metrics/family_*/*.rs   `if speed >= N` gates -> which column
                                                        emits 0 from which level up
  live_delta_calc_modular/src/main.rs                 HELP table -> the documented count of
                                                        computed columns per level

The gates are parsed from the Rust with pattern-specific regexes (the four
shapes that occur are listed in _gates_in_file). Every local identifier the
gates mention must resolve to a csv_header() column, either directly or
through a `column: ident` struct-literal line in the same file; an unresolved
identifier is an error, never a silent omission.

The parsed table is then checked against the HELP table's per-level counts.
If they disagree the roster refuses to build -- that disagreement is the
event worth being told about (Dispute 4 in plan10_analysis_console_proposal.md).

Family and submodule come from the Python taxonomy the capture console
already imports (plan07_campaign/subset_run.py), compared to the Rust BY NAME,
never by position: the two files do not share an order.

Run:  python3 plan10_analysis/channel_roster.py [--out roster.json] [--speed N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # .../plan10_analysis
QEMU_DIR = HERE.parent                            # .../VM_Capture_QEMU
REPO = QEMU_DIR.parent.parent                     # .../mem_sig
RUST_SRC = REPO / "VM_sampler" / "VM_Capture" / "live_delta_calc_modular" / "src"
MOD_RS = RUST_SRC / "metrics" / "mod.rs"
MAIN_RS = RUST_SRC / "main.rs"
CAMPAIGN = QEMU_DIR / "plan07_campaign"

SCHEMA = "plan10.channel_roster.v1"


class RosterError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------

def csv_header_names(mod_rs: Path = MOD_RS) -> list[str]:
    """The column names csv_header() returns, in the order the differ writes them."""
    text = mod_rs.read_text()
    m = re.search(r'pub fn csv_header\(\)[^{]*\{\s*"([^"]+)"', text)
    if not m:
        raise RosterError(f"csv_header() string not found in {mod_rs}")
    names = [n.strip() for n in m.group(1).split(",") if n.strip()]
    if len(set(names)) != len(names):
        raise RosterError("csv_header() has duplicate names")
    return names


# ---------------------------------------------------------------------------
# speed gates
# ---------------------------------------------------------------------------

_RE_INLINE = re.compile(r"(\w+):\s*if speed >= (\d)\s*\{\s*0(?:\.0)?\s*\}")
_RE_LET_ONE = re.compile(r"let (\w+) = if speed >= (\d)\s*\{\s*0(?:\.0)?\s*\}")
_RE_LET_TUPLE = re.compile(r"let \(([^)]+)\) = if speed >= (\d)\s*\{\s*\((.*?)\)\s*\}", re.S)
_RE_RETURN_DEFAULT = re.compile(r"if speed >= (\d)\s*\{[^}]*?return (\w+)::default\(\);")
_RE_STRUCT_FIELDS = re.compile(r"pub struct (\w+)\s*\{([^}]*)\}", re.S)
_RE_FIELD = re.compile(r"pub (\w+):")


def _split_top_level(s: str) -> list[str]:
    """Split a tuple body on commas that are not inside parentheses.

    `(spearman_fast(p, q, &sh.hp, &sh.hq), 0.0)` is two values, not five.
    """
    out, depth, cur = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


def _gates_in_file(path: Path) -> list[tuple[str, int]]:
    """(local identifier, speed level at which it is zeroed) for every gate in one file.

    The four shapes the family sources use:
      a. `field: if speed >= N { 0.0 } else { ... }`         (family_c.rs, texture.rs)
      b. `let x = if speed >= N { 0.0 } else { ... }`        (informational.rs)
      c. `let (a, b) = if speed >= N { (0.0, expr) } else`   (informational.rs, structure.rs,
                                                              texture.rs) -- only the tuple
                                                              positions that are 0 are dropped
      d. `if speed >= N { return Struct::default(); }`       (spatial_shift.rs) -- every field
    """
    text = path.read_text()
    out: list[tuple[str, int]] = []
    for ident, lvl in _RE_INLINE.findall(text):
        out.append((ident, int(lvl)))
    for ident, lvl in _RE_LET_ONE.findall(text):
        out.append((ident, int(lvl)))
    for idents, lvl, vals in _RE_LET_TUPLE.findall(text):
        names = [i.strip() for i in idents.split(",")]
        values = _split_top_level(vals)
        if len(names) != len(values):
            raise RosterError(f"{path.name}: tuple gate arity mismatch: {idents} vs {vals}")
        for n, v in zip(names, values):
            if re.fullmatch(r"0(\.0)?", v):
                out.append((n, int(lvl)))
    for lvl, struct in _RE_RETURN_DEFAULT.findall(text):
        m = next((s for s in _RE_STRUCT_FIELDS.finditer(text) if s.group(1) == struct), None)
        if not m:
            raise RosterError(f"{path.name}: `return {struct}::default()` but struct not found in file")
        for field in _RE_FIELD.findall(m.group(2)):
            out.append((field, int(lvl)))
    return out


def _resolve(ident: str, names: set[str], file_text: str, path: Path) -> str:
    """Map a local identifier to a csv_header() column, or fail loudly."""
    if ident in names:
        return ident
    # struct-literal line `column: ident as f32` / `column: ident,`
    for m in re.finditer(r"\b(\w+):\s*\(?\s*" + re.escape(ident) + r"\b", file_text):
        if m.group(1) in names:
            return m.group(1)
    raise RosterError(f"{path.name}: gated identifier {ident!r} does not resolve to a column")


def speed_drop_levels(names: list[str], rust_src: Path = RUST_SRC) -> dict[str, int]:
    """column -> lowest speed level at which the differ emits 0 for it."""
    nameset = set(names)
    drop: dict[str, int] = {}
    files = sorted((rust_src / "metrics").rglob("*.rs"))
    files = [f for f in files if f.name != "mod.rs"]
    for f in files:
        text = f.read_text()
        for ident, lvl in _gates_in_file(f):
            col = _resolve(ident, nameset, text, f)
            drop[col] = min(lvl, drop.get(col, 99))
    return drop


# ---------------------------------------------------------------------------
# the documented level table (main.rs HELP)
# ---------------------------------------------------------------------------

_RE_LEVEL_ROW = re.compile(r"^\s+(\d)\s+(\d+)\s+~([\d.]+) s\s+(.*?)\s*$", re.M)


def help_levels(main_rs: Path = MAIN_RS) -> list[dict]:
    text = main_rs.read_text()
    rows = []
    for lvl, n, secs, desc in _RE_LEVEL_ROW.findall(text):
        rows.append({"speed": int(lvl), "computed": int(n), "seconds_per_pair": float(secs),
                     "drops": desc.strip()})
    if [r["speed"] for r in rows] != [0, 1, 2, 3, 4]:
        raise RosterError(f"HELP level table not found or malformed in {main_rs}")
    return rows


# ---------------------------------------------------------------------------
# the Python taxonomy (by name)
# ---------------------------------------------------------------------------

def taxonomy() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    if str(CAMPAIGN) not in sys.path:
        sys.path.insert(0, str(CAMPAIGN))
    from subset_run import FEATURES_BY_GROUP, SUBMODULE_COLUMNS  # noqa: E402
    return FEATURES_BY_GROUP, SUBMODULE_COLUMNS


# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def build_roster() -> dict:
    names = csv_header_names()
    drop = speed_drop_levels(names)
    levels = help_levels()
    groups, subs = taxonomy()

    # documented vs parsed: the check that makes this a derivation and not a guess
    for row in levels:
        alive = sum(1 for n in names if drop.get(n, 99) > row["speed"])
        if alive != row["computed"]:
            dead = sorted(n for n in names if drop.get(n, 99) <= row["speed"])
            raise RosterError(
                f"speed {row['speed']}: parsed gates leave {alive} columns computed, "
                f"HELP says {row['computed']}. parsed dead set: {dead}")

    # taxonomy vs rust, by name
    tax_names = [c for g in groups.values() for c in g]
    missing = sorted(set(names) - set(tax_names))
    extra = sorted(set(tax_names) - set(names))
    if missing or extra:
        raise RosterError(f"taxonomy/rust name mismatch: rust-only={missing} python-only={extra}")
    group_of = {c: g for g, cols in groups.items() for c in cols}
    sub_of = {c: s for s, cols in subs.items() for c in cols}

    channels = []
    for i, n in enumerate(names):
        channels.append({"name": n, "rust_index": i, "group": group_of[n],
                         "submodule": sub_of.get(n), "drop_at": drop.get(n)})

    # the level-4 line names "struct_entropy", which is not a column: record what it resolved to
    l4 = sorted(n for n in names if drop.get(n) == 4)

    return {
        "schema": SCHEMA,
        "n_total": len(names),
        "channels": channels,
        "levels": levels,
        "level4_struct_entropy_resolves_to": l4,
        "derivation": {
            "method": "parsed",
            "names_from": str(MOD_RS.relative_to(REPO)),
            "gates_from": [str(p.relative_to(REPO)) for p in sorted((RUST_SRC / "metrics").rglob("*.rs")) if p.name != "mod.rs"],
            "counts_checked_against": str(MAIN_RS.relative_to(REPO)),
            "taxonomy_from": "VM_sampler/VM_Capture_QEMU/plan07_campaign/subset_run.py",
            "source_sha256_16": {str(MOD_RS.relative_to(REPO)): _sha(MOD_RS), str(MAIN_RS.relative_to(REPO)): _sha(MAIN_RS)},
        },
    }


def live_at(roster: dict, speed: int) -> list[str]:
    return [c["name"] for c in roster["channels"] if c["drop_at"] is None or c["drop_at"] > speed]


def dead_at(roster: dict, speed: int) -> list[str]:
    return [c["name"] for c in roster["channels"] if c["drop_at"] is not None and c["drop_at"] <= speed]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=None, help="write roster JSON here")
    ap.add_argument("--speed", type=int, default=None, help="print live/dead at this level")
    a = ap.parse_args()
    try:
        r = build_roster()
    except RosterError as e:
        print(f"[channel_roster] REFUSED: {e}", file=sys.stderr)
        return 1
    if a.out:
        a.out.write_text(json.dumps(r, indent=1) + "\n")
        print(f"[channel_roster] wrote {a.out}")
    print(f"[channel_roster] {r['n_total']} columns; per level computed: "
          + ", ".join(f"{l['speed']}:{l['computed']}" for l in r["levels"]))
    print(f"[channel_roster] level-4 'struct_entropy' resolves to: {r['level4_struct_entropy_resolves_to']}")
    if a.speed is not None:
        print(f"[channel_roster] speed {a.speed}: {len(live_at(r, a.speed))} live, dead = {dead_at(r, a.speed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
