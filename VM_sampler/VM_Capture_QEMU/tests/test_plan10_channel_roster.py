#!/usr/bin/env python3
"""channel_roster.py: the 64 columns and their speed drop levels, derived from the Rust.

The load-bearing checks: the parsed gates reconcile with the differ's own
HELP table at every level (64/63/51/50/48), the level-4 line "struct_entropy"
resolves to real columns, and the Python taxonomy agrees with the Rust BY NAME
while its order differs (so a positional check would lie).

Run:  python3 tests/test_plan10_channel_roster.py      (plain asserts, no pytest dep)
      pytest tests/test_plan10_channel_roster.py
"""
from __future__ import annotations

import sys
from pathlib import Path

QEMU_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QEMU_DIR))
from plan10_analysis import channel_roster as cr  # noqa: E402


def test_split_top_level():
    assert cr._split_top_level("spearman_fast(p, q, &sh.hp, &sh.hq), 0.0") == ["spearman_fast(p, q, &sh.hp, &sh.hq)", "0.0"]
    assert cr._split_top_level("0.0, 0.0, 0.0, 0.0") == ["0.0"] * 4


def test_roster_reconciles_with_help_table():
    r = cr.build_roster()
    assert r["n_total"] == 64
    assert [l["computed"] for l in r["levels"]] == [64, 63, 51, 50, 48]
    for l in r["levels"]:
        assert len(cr.live_at(r, l["speed"])) == l["computed"]


def test_level4_struct_entropy_resolves_to_two_columns():
    r = cr.build_roster()
    assert sorted(r["level4_struct_entropy_resolves_to"]) == ["struct_ent_change", "struct_ent_q"]


def test_dead_at_speed_2_is_the_heavy_twelve_plus_lz():
    r = cr.build_roster()
    dead = set(cr.dead_at(r, 2))
    assert len(dead) == 13
    for name in ("ncd", "lz_change", "kendall", "cross_corr_lag", "phase_corr", "byte_rotation",
                 "bigram_ent", "autocorr_peak", "glcm_contrast", "glcm_homogeneity", "glcm_energy",
                 "glcm_correlation", "high_freq_frac"):
        assert name in dead, name
    assert cr.dead_at(r, 0) == []
    assert "csize_delta" in cr.dead_at(r, 3) and "csize_delta" not in cr.dead_at(r, 2)


def test_taxonomy_matches_by_name_not_by_position():
    r = cr.build_roster()
    groups, subs = cr.taxonomy()
    rust_order = [c["name"] for c in r["channels"]]
    tax_order = [c for g in groups.values() for c in g]
    assert set(rust_order) == set(tax_order)
    assert rust_order != tax_order, "orders agree today; the by-name comparison is still the right one"
    assert all(c["group"] for c in r["channels"])
    assert sum(1 for c in r["channels"] if c["group"] == "amount") == 20


def test_derivation_is_recorded():
    r = cr.build_roster()
    assert r["derivation"]["method"] == "parsed"
    assert r["derivation"]["names_from"].endswith("metrics/mod.rs")
    assert len(r["derivation"]["gates_from"]) >= 5


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
