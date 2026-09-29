#!/usr/bin/env python3
"""The content-change reading on the canvas: Persistent pages and Collapse by median or quantile.

The encoding paper (p2e_skeleton.tex, Sec. II) computes the content-change reading on the pages
that changed in pair t and change again in pair t+1, the set S_t & S_{t+1}, "summarised per pair
by its median and by quantiles fixed in advance". plan11's extract computes the same thing as its
`r_*_qNN_per` columns. This file checks the canvas pieces on known inputs, and that the canvas
and plan11 give the same numbers on the same rows.

Run:  python3 tests/test_plan10_content_change.py
      pytest tests/test_plan10_content_change.py
"""
from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

import numpy as np

QEMU_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QEMU_DIR))
from plan10_analysis import channel_roster, corpus_manifest, scheme as S   # noqa: E402
from plan10_analysis.modules import build_modules, collapsed_channel_names, quantile_tag   # noqa: E402
from plan10_analysis.runner import stages                                  # noqa: E402

QS = (0.05, 0.25, 0.50, 0.75, 0.95)


def _field():
    """Three pairs with the page sets {3,4,5}, {3,9}, {3,9,7}; each row's value is 10*seq + page,
    so any row can be recognised after filtering."""
    seq = np.array([1, 1, 1, 2, 2, 3, 3, 3], dtype=np.int32)
    page = np.array([3, 4, 5, 3, 9, 3, 9, 7], dtype=np.int32)
    val = (10 * seq + page).astype(np.float32)
    return {"seq": seq, "page_index": page, "cols": {"hamming": val}, "z": None, "channels": ["hamming"],
            "n_pairs": 3, "n_pages": 100, "block": None}


def _rows(f):
    return sorted(zip(f["seq"].tolist(), f["page_index"].tolist(), f["cols"]["hamming"].tolist()))


# ---------------------------------------------------------------------------------- Persistent pages

def test_persistent_pages_keeps_the_pages_that_change_again():
    f = _field()
    # S1 & S2 = {3}; S2 & S3 = {3, 9}; pair 3 has no partner and takes pair 2's rows
    p = stages.persistent_pages(f, lag=1, side="t", edge="replicate")
    assert _rows(p) == [(1, 3, 13.0), (2, 3, 23.0), (2, 9, 29.0), (3, 3, 23.0), (3, 9, 29.0)]
    assert p["n_pairs"] == 3 and p["channels"] == ["hamming"]
    assert p["persistent"] == {"lag": 1, "side": "t", "edge": "replicate"}
    e = stages.persistent_pages(f, lag=1, side="t", edge="empty")
    assert _rows(e) == [(1, 3, 13.0), (2, 3, 23.0), (2, 9, 29.0)]
    # side t+lag: the later pair's values, relabelled to pair t
    later = stages.persistent_pages(f, lag=1, side="t+lag", edge="empty")
    assert _rows(later) == [(1, 3, 23.0), (2, 3, 33.0), (2, 9, 39.0)]
    # lag 2: S1 & S3 = {3}
    assert _rows(stages.persistent_pages(f, lag=2, edge="empty")) == [(1, 3, 13.0)]
    # the input is not modified
    assert _rows(f) == _rows(_field())


def test_persistent_pages_refuses_what_it_cannot_do():
    f = _field()
    for kw in ({"lag": 0}, {"lag": 3}, {"side": "t+1"}, {"edge": "drop"}):
        try:
            stages.persistent_pages(f, **kw)
        except ValueError:
            continue
        raise AssertionError(f"accepted {kw}")


def test_persistent_pages_on_blocked_and_complex_fields():
    f = _field()
    # a blocked field: membership is by page, the block column rides along with its row
    b = dict(f, block=np.array([0, 0, 0, 0, 1, 0, 1, 1], dtype=np.int32), block_w=8, block_h=8, n_blocks=2)
    pb = stages.persistent_pages(b, edge="empty")
    got = sorted(zip(pb["seq"].tolist(), pb["page_index"].tolist(), pb["block"].tolist()))
    assert got == [(1, 3, 0), (2, 3, 0), (2, 9, 1)]
    # a complex field: z is filtered with the rows
    z = dict(f, cols=None, z=(f["cols"]["hamming"] * (1 + 1j)).astype(np.complex64))
    pz = stages.persistent_pages(z, edge="empty")
    assert sorted(pz["z"].real.tolist()) == [13.0, 23.0, 29.0]


# ---------------------------------------------------------------------------------- Collapse by quantile

def test_quantile_with_zeros_is_numpy_quantile():
    rng = np.random.default_rng(7)
    for _ in range(300):
        v = rng.normal(size=int(rng.integers(0, 40))) * rng.choice([1, 100])
        z = int(rng.integers(0, 60))
        q = float(rng.choice([0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0, rng.random()]))
        got = stages._quantile_with_zeros(v, z, q)
        ref = float(np.quantile(np.concatenate([v, np.zeros(z)]), q)) if v.size + z else 0.0
        assert got == ref, (v, z, q, got, ref)


def test_collapse_median_and_quantile_on_known_inputs():
    f = _field()
    # excluded: over each pair's rows. pair 1 {13,14,15}, pair 2 {23,29}, pair 3 {33,39,37}
    m = stages.collapse(f, "excluded", "median")
    assert np.allclose(m["values"], [14, 26, 37]) and m["channels"] == ["hamming_q50"]
    q = stages.collapse(f, "excluded", "quantile", 0.25)
    assert np.allclose(q["values"], [np.quantile([13, 14, 15], .25), np.quantile([23, 29], .25), np.quantile([33, 37, 39], .25)])
    assert q["channels"] == ["hamming_q25"]
    # zero: every one of the 100 pages counts, unchanged ones as 0
    zz = stages.collapse(f, "zero", "quantile", 0.99)
    ref = [np.quantile(np.concatenate([[13, 14, 15], np.zeros(97)]), .99),
           np.quantile(np.concatenate([[23, 29], np.zeros(98)]), .99),
           np.quantile(np.concatenate([[33, 39, 37], np.zeros(97)]), .99)]
    assert np.allclose(zz["values"], ref)
    assert np.allclose(stages.collapse(f, "zero", "median")["values"], [0, 0, 0])
    # a pair with no rows is 0 under excluded, as the mean is
    g = dict(f, n_pairs=4)
    assert np.allclose(stages.collapse(g, "excluded", "median")["values"], [14, 26, 37, 0])
    # several channels: one column each
    h = dict(f, cols={"hamming": f["cols"]["hamming"], "l0": np.arange(8, dtype=np.float32)}, channels=["hamming", "l0"])
    mh = stages.collapse(h, "excluded", "median")
    assert mh["values"].shape == (3, 2) and mh["channels"] == ["hamming_q50", "l0_q50"]
    assert np.allclose(mh["values"][:, 1], [1, 3.5, 6])


def test_collapse_names_the_population_and_leaves_the_old_reductions_alone():
    f = _field()
    p = stages.persistent_pages(f, edge="empty")
    assert stages.collapse(p, "excluded", "median")["channels"] == ["hamming_q50_per"]
    assert stages.collapse(p, "excluded", "mean")["channels"] == ["hamming_per"]
    assert stages.collapse(p, "zero", "changed_fraction")["channels"] == ["changed_fraction_per"]
    # an ordinary field: mean and K/N are named as before this change
    assert stages.collapse(f, "zero", "mean")["channels"] == ["hamming"]
    assert stages.collapse(f, "zero", "changed_fraction")["channels"] == ["changed_fraction"]
    assert collapsed_channel_names(["x"], "quantile", 0.05) == ["x_q05"] and quantile_tag(0.95) == "q95"


def test_collapse_refuses_a_quantile_it_cannot_take():
    f = _field()
    z = dict(f, cols=None, z=np.ones(8, dtype=np.complex64))
    for args in ((z, "excluded", "median"), (f, "excluded", "quantile", 1.5), (f, "excluded", "mode")):
        try:
            stages.collapse(*args)
        except ValueError:
            continue
        raise AssertionError(f"accepted {args[1:]}")


# ---------------------------------------------------------------------------------- canvas = plan11

def _snapshots_and_field(seed=11, T=7, n_pages=400):
    """Random changed pages with integer channel values, every pair sharing a core of pages so that
    no persistent set is empty; returned as plan11 Snapshots and as the canvas's field."""
    from plan11_encoding_ladder import extract as X
    rng = np.random.default_rng(seed)
    core = rng.choice(n_pages, 40, replace=False)
    snaps, seq, page, ham, l0, l1 = [], [], [], [], [], []
    for t in range(1, T + 1):
        extra = rng.choice(n_pages, int(rng.integers(20, 120)), replace=False)
        pg = np.unique(np.concatenate([core[rng.random(core.size) < 0.8], extra]))
        a = rng.integers(1, 4097, size=pg.size)                       # l0 >= 1 on a changed page
        h = np.minimum(a * 8, rng.integers(1, 9, size=pg.size) * a)   # 1 to 8 bits per changed byte
        b = a * rng.integers(1, 256, size=pg.size)                    # l1 >= l0
        snap, _ = X.finalize_buffer(t, pg.tolist(), h.tolist(), a.tolist(), b.tolist())
        snaps.append(snap)
        seq += [t] * pg.size; page += pg.tolist(); ham += h.tolist(); l0 += a.tolist(); l1 += b.tolist()
    field = {"seq": np.array(seq, np.int32), "page_index": np.array(page, np.int32),
             "cols": {"hamming": np.array(ham, np.float32), "l0": np.array(l0, np.float32), "l1": np.array(l1, np.float32)},
             "z": None, "channels": ["hamming", "l0", "l1"], "n_pairs": T, "n_pages": n_pages, "block": None}
    return snaps, field


def test_canvas_content_change_equals_plan11():
    """Persistent pages, then Ratios, then Collapse by quantile, against plan11's r_*_qNN_per on the
    same rows, for both sides and every quantile the paper fixes. The canvas carries float32, so the
    two agree to float32 precision, not to the bit."""
    from plan11_encoding_ladder import extract as X, schema as SC
    snaps, field = _snapshots_and_field()
    cols = list(SC.EXTRACT_COLUMNS)
    T = field["n_pairs"]
    specs = ["l0/page", "l1/l0", "hamming/l0"]
    prefixes = ["r_l0", "r_l1l0", "r_haml0"]
    for side, tk_side in (("t", "t"), ("t+lag", "t+1")):
        per = stages.persistent_pages(field, lag=1, side=side, edge="empty")
        rat = stages.ratios(per, specs, 4096)
        # the filter and the ratios commute: Ratios first gives the same rows
        rat2 = stages.persistent_pages(stages.ratios(field, specs, 4096), lag=1, side=side, edge="empty")
        for q in QS:
            red = "median" if q == 0.5 else "quantile"
            canvas = stages.collapse(rat, "excluded", red, q)
            assert canvas["channels"] == [f"{n}_{quantile_tag(q)}_per" for n in ("l0_over_page", "l1_over_l0", "hamming_over_l0")]
            assert np.array_equal(canvas["values"], stages.collapse(rat2, "excluded", red, q)["values"])
            for t in range(1, T):
                tk = X.row_values(snaps[t - 1], snaps[t], n_pages=field["n_pages"], quantiles=QS, persist_side=tk_side, page_size=4096)
                want = [tk[cols.index(f"{p}_{quantile_tag(q)}_per")] for p in prefixes]
                assert np.allclose(canvas["values"][t - 1], want, rtol=1e-6, atol=0), (side, q, t, canvas["values"][t - 1], want)
            # the last pair has no partner: plan11 leaves it blank, edge=empty gives it 0
            assert np.all(canvas["values"][T - 1] == 0)


def test_replicate_repeats_the_last_partnered_value():
    snaps, field = _snapshots_and_field(seed=3)
    rat = stages.ratios(stages.persistent_pages(field, lag=1, edge="replicate"), ["l1/l0"], 4096)
    v = stages.collapse(rat, "excluded", "median")["values"]
    assert v[-1] == v[-2] and len(v) == field["n_pairs"]


# ---------------------------------------------------------------------------------- the scheme

def _ctx():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "zstd_local"
        for i, wl in enumerate(["mem_alpha_v2", "mem_alpha_v2", "cpu_beta_v2"]):
            d = root / wl.split("_", 1)[0] / wl / f"args_--duration_450_--seed_{i}" / f"rep00{i + 1}__t"
            d.mkdir(parents=True)
            for j in range(701):
                (d / f"{j:06d}.zst").write_bytes(b"x")
        manifest = corpus_manifest.scan(root)
    for r in manifest["recordings"]:
        r["has"]["substrate_csv"] = True
    return S.Context(channel_roster.build_roster(), manifest, build_modules(), S.load_config())


def _content_scheme(ctx):
    """The b1 example with Channels -> Persistent pages -> Ratios -> Collapse(median, excluded)."""
    s = copy.deepcopy(S.make_examples(ctx.manifest)["b1"])
    ch = next(n for n in s["nodes"] if n["module"] == "channels")
    col = next(n for n in s["nodes"] if n["module"] == "collapse")
    ch["params"]["chans"] = ["l0", "l1", "hamming"]
    s["nodes"] += [dict(col, id="n_p", module="persist_pages", params={"lag": 1, "side": "t", "edge": "replicate"}),
                   dict(col, id="n_r", module="ratios", params={"ratios": ["l0/page", "l1/l0", "hamming/l0"], "page_bytes": 4096})]
    col["params"] = {"reduce": "median", "unchanged": "excluded", "q": 0.5}
    pipe = next(pp for pp in s["pipes"] if pp["from"][0] == ch["id"])
    pipe["from"] = ["n_r", "out"]
    s["pipes"] += [{"from": [ch["id"], "field"], "to": ["n_p", "in"]}, {"from": ["n_p", "out"], "to": ["n_r", "in"]}]
    return s, col["id"]


def test_scheme_types_the_content_change_branch():
    ctx = _ctx()
    s, cid = _content_scheme(ctx)
    issues = S.validate(s, ctx)
    assert not [i for i in issues if i["sev"] == "hard"], issues
    assert any(i["sev"] == "note" and i["node"] == "n_p" and "no partner" in i["msg"] for i in issues)
    g = S.Graph(s, ctx)
    p = S.descriptor(g, "n_p", {})
    assert p["type"] == "field" and p["persistent"] is True and p["channels"] == ["l0", "l1", "hamming"]
    d = S.descriptor(g, cid, {})
    assert d["type"] == "series" and d["complex"] is False
    assert d["channels"] == ["l0_over_page_q50_per", "l1_over_l0_q50_per", "hamming_over_l0_q50_per"]


def test_scheme_refuses_and_warns():
    ctx = _ctx()
    s, cid = _content_scheme(ctx)
    col = next(n for n in s["nodes"] if n["id"] == cid)
    col["params"] = {"reduce": "quantile", "unchanged": "excluded", "q": 1.5}
    assert any(i["sev"] == "hard" and i["node"] == cid and "between 0 and 1" in i["msg"] for i in S.validate(s, ctx))
    col["params"] = {"reduce": "median", "unchanged": "zero", "q": 0.5}
    assert any(i["sev"] == "note" and i["node"] == cid and "every page of the dump" in i["msg"] for i in S.validate(s, ctx))
    col["params"] = {"reduce": "changed_fraction", "unchanged": "zero"}
    assert any(i["sev"] == "note" and i["node"] == cid and "not APF" in i["msg"] for i in S.validate(s, ctx))
    next(n for n in s["nodes"] if n["id"] == "n_p")["params"]["lag"] = 0
    assert any(i["sev"] == "hard" and i["node"] == "n_p" and "at least 1" in i["msg"] for i in S.validate(s, ctx))


def test_the_catalogue_offers_both():
    m = {x["id"]: x for x in build_modules()["modules"]}
    assert m["persist_pages"]["tier"] == "divide"
    assert [q["k"] for q in m["persist_pages"]["params"]] == ["lag", "side", "edge"]
    assert [o[0] for o in m["collapse"]["params"][0]["opts"]] == ["mean", "changed_fraction", "median", "quantile"]
    assert [q["k"] for q in m["collapse"]["params"]] == ["reduce", "unchanged", "q"]


def test_the_browser_knows_every_module_that_reshapes_a_signal():
    """The canvas checks a scheme twice: scheme.py on the bridge, and its own copy of the rules in
    the page (descOf). A source, compose or divide module changes the type or the channels of what
    flows, so the page needs its own case for it; without one the page sees nothing leave the node
    and refuses what feeds from it (Persistent pages shipped first without one: Ratios below it
    read "carries nothing"). Lens and output modules share the page's generic handling."""
    tpl = (QEMU_DIR / "plan10_analysis" / "ui" / "analysis_console.template.html").read_text()
    start = tpl.index("function descOf(")
    body = tpl[start:tpl.index("\nfunction ", start + 10)]
    reshaping = [m["id"] for m in build_modules()["modules"] if m["tier"] in ("source", "compose", "divide")]
    missing = [i for i in reshaping if f'case "{i}"' not in body]
    assert not missing, f"no case in the page's descOf for {missing}; mirror scheme.py's propagation there"
    # and the two new ones name their channels the way Python does
    assert "collapsedNames(" in body and 'case "persist_pages"' in body


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
