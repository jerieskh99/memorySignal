#!/usr/bin/env python3
"""executor.py: the three examples run end to end on a synthetic corpus and write what they promise.

Also checks the stages on known inputs: Collapse in changed_fraction mode reproduces the
fixture's APF exactly, Window tiles have the arithmetic the console shows, and a stop
written to control.json is honoured.

Needs the differ binary and zstd; skips with a message otherwise.

Run:  python3 tests/test_plan10_runner_executor.py
      pytest tests/test_plan10_runner_executor.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

QEMU_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QEMU_DIR))
from plan10_analysis import corpus_manifest, scheme as S       # noqa: E402
from plan10_analysis.modules import build_modules              # noqa: E402
from plan10_analysis.runner import chain, differ, executor, stages  # noqa: E402
from plan10_analysis.testing import synth                        # noqa: E402


def _have_tools() -> bool:
    try:
        differ.find_differ()
    except differ.DifferError as e:
        print(f"skip: {e}")
        return False
    if not chain.zstd_available():
        print("skip: zstd not on PATH")
        return False
    return True


def _corpus(td: Path, n_snapshots=10):
    root = td / "corpus"
    changes = synth.make_corpus(root, n_snapshots=n_snapshots)
    manifest = corpus_manifest.scan(root)
    mp = td / "manifest.json"
    mp.write_text(json.dumps(manifest))
    return root, manifest, mp, changes


def _example(manifest, k, **cells_over):
    s = S.make_examples(manifest)[k]
    cells = next(n for n in s["nodes"] if n["module"] == "cells")
    cells["params"].update(min_pairs=1, **cells_over)
    cells["params"]["sel"] = [r["id"] for r in manifest["recordings"]]
    s["acknowledged"] = [{"id": "no_substrate", "at": "test", "note": "synthetic chains"}]
    return s


def test_stages_on_known_inputs():
    seq = np.array([1, 1, 1, 2, 2, 3], dtype=np.int32)
    field = {"seq": seq, "page_index": np.array([3, 4, 5, 3, 9, 7], dtype=np.int32),
             "cols": {"hamming": np.array([8, 16, 4, 2, 2, 100], dtype=np.float32)}, "z": None,
             "channels": ["hamming"], "n_pairs": 3, "n_pages": 100, "block": None}
    s = stages.collapse(field, "zero", "changed_fraction")
    assert np.allclose(s["values"], [3 / 100, 2 / 100, 1 / 100]) and s["channels"] == ["changed_fraction"]
    m = stages.collapse(field, "zero", "mean")
    assert np.allclose(m["values"], [28 / 100, 4 / 100, 100 / 100])
    e = stages.collapse(field, "excluded", "mean")
    assert np.allclose(e["values"], [28 / 3, 4 / 2, 100 / 1])
    t = stages.window({"values": np.arange(10, dtype=np.float32), "channels": ["x"], "complex": False, "block": None, "n_pages": 1}, 4, 2)
    assert t["X"].shape == (4, 4) and t["keys"][0] == (None, 0, 1) and t["keys"][-1] == (None, 3, 7)
    tz = stages.window({"values": np.arange(10, dtype=np.float32), "channels": ["x"], "complex": False, "block": None, "n_pages": 1}, 4, 4, edge="zero")
    assert tz["X"].shape == (3, 4) and tz["X"][-1].tolist() == [8, 9, 0, 0]
    f = stages.stats(t, ["mean", "max", "duty"])
    assert f["names"] == ["mean", "max", "duty"] and f["rows"].shape == (4, 3) and f["rows"][0, 1] == 3
    # complex on matching rows; phase conventions
    dirf = dict(field, cols={"cosine": np.array([0, 1, 0.5, 0, 1, 0.5], dtype=np.float32)}, channels=["cosine"])
    z = stages.complex_field(field, dirf, "pi")
    assert z["z"] is not None and np.isclose(np.angle(z["z"][1]), np.pi)
    z2 = stages.complex_field(field, dirf, "2pi")
    assert np.isclose(np.angle(z2["z"][1]), 0.0, atol=1e-6)   # the collision: distance 1 lands on angle 0


def test_blocks_overlapping_tiling_and_gapped():
    """Membership against brute force, row replication, and one block count everywhere."""
    pages = np.array([0, 3, 4, 7, 8, 11, 15, 19], dtype=np.int32)
    f = {"seq": np.ones(len(pages), np.int32), "page_index": pages,
         "cols": {"hamming": pages.astype(np.float32)}, "z": None, "channels": ["hamming"],
         "n_pairs": 1, "n_pages": 20, "block": None}
    for wp, hp in ((4, 4), (8, 4), (8, 2), (8, 8), (8, 12), (5, 5), (6, 4), (20, 20), (20, 1)):
        nb = stages.n_blocks(20, wp, hp)
        b = stages.block(f, wp, hp)
        assert b["n_blocks"] == nb and b["block_w"] == wp and b["block_h"] == hp
        want = {int(p): [i for i in range(nb) if i * hp <= p < i * hp + wp] for p in pages}
        got: dict = {}
        for p_, blk in zip(b["page_index"], b["block"]):
            got.setdefault(int(p_), []).append(int(blk))
        for p in want:
            assert sorted(got.get(p, [])) == want[p], (wp, hp, p, got.get(p), want[p])
        # every row keeps its own value through the replication
        assert np.array_equal(b["cols"]["hamming"], b["page_index"].astype(np.float32))
        assert len(b["page_index"]) == sum(len(v) for v in want.values())
    # overlap replicates, a tiling does not, a gap drops
    assert len(stages.block(f, 8, 4)["page_index"]) == 13
    assert len(stages.block(f, 4, 4)["page_index"]) == 8
    assert len(stages.block(f, 8, 12)["page_index"]) == 6
    # whole blocks only, the same rule Window's edge=drop uses in time
    assert stages.n_blocks(20, 8, 8) == 2 and stages.n_blocks(20, 4, 4) == 5
    assert stages.n_blocks(20, 21, 1) == 0
    try:
        stages.block(f, 21, 1)
        assert False
    except ValueError as e:
        assert "does not fit" in str(e)
    for bad in ((0, 4), (4, 0)):
        try:
            stages.n_blocks(20, *bad)
            assert False, bad
        except ValueError:
            pass
    # collapse reduces each block over the block's own width
    for s in stages.collapse(stages.block(f, 8, 4), reduce="changed_fraction"):
        lo = s["block"] * 4
        assert np.isclose(float(s["values"][0]), sum(1 for p in pages if lo <= p < lo + 8) / 8)
    # the runner's block count is the one scheme.py estimates from
    for n_pages, wp, hp in ((20, 8, 4), (1024, 300, 300), (262144, 8192, 4096)):
        assert stages.n_blocks(n_pages, wp, hp) == (n_pages - wp) // hp + 1


def test_blocked_tile_count_matches_the_estimate():
    """The estimate must carry the block factor through Collapse; it used to drop it entirely."""
    if not _have_tools():
        return
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        root, manifest, mp, _ = _corpus(td, n_snapshots=30)
        ctx = S.Context(__import__("plan10_analysis.channel_roster", fromlist=["x"]).build_roster(),
                        manifest, build_modules(), S.load_config())

        def scheme_with(wp, hp, label):
            n = lambda i, m, **p: {"id": i, "module": m, "params": p, "x": 0, "y": 0}
            e = lambda a, ap, b, bp: {"from": [a, ap], "to": [b, bp]}
            return {"schema": "plan10.scheme.v1", "label": label, "acknowledged": [{"id": "no_substrate", "at": "t"}],
                    "nodes": [n("c", "cells", sel=[r["id"] for r in manifest["recordings"]], min_pairs=1),
                              n("ch", "channels", chans=["hamming"]), n("b", "block", wp=wp, hp=hp),
                              n("co", "collapse", reduce="changed_fraction"), n("wi", "window", w=8, h=4),
                              n("st", "stats", feats=["mean"]), n("w", "write")],
                    "pipes": [e("c", "cells", "ch", "cells"), e("ch", "field", "b", "in"), e("b", "out", "co", "in"),
                              e("co", "out", "wi", "in"), e("wi", "out", "st", "in"), e("st", "out", "w", "in")]}

        cfg_pages = ctx.config["n_pages_default"]
        for wp, hp in ((256, 256), (256, 128), (256, 64), (128, 256)):
            sch = scheme_with(wp, hp, f"blk_{wp}_{hp}")
            est = S.estimate(sch, ctx)
            windows = est["windows"]
            # the estimate counts blocks over the CONFIG page count, and says so in a note
            assert est["tiles"] == windows * stages.n_blocks(cfg_pages, wp, hp), (wp, hp, est)
            assert any("assume" in i["msg"] and "pages per dump" in i["msg"] for i in S.validate(sch, ctx))
            # the runner counts them over the recording's ACTUAL page count
            sp = td / f"{sch['label']}.json"
            sp.write_text(json.dumps(sch))
            assert executor.run(sp, td / "out", {"kind": "local", "root": str(root)}, td / "l1",
                                speed=2, manifest_path=mp, acknowledge_all=True) == 0
            z = np.load(td / "out" / sch["label"] / "features.npz")
            assert z["X"].shape[0] == windows * stages.n_blocks(synth.N_PAGES, wp, hp), (wp, hp, z["X"].shape)
            assert len(set(z["tile_keys"]["block"].tolist())) == stages.n_blocks(synth.N_PAGES, wp, hp)


def test_full_page_resolution_tiles():
    """The page-by-time image: dense materialisation, its guards, and the median reduction."""
    seq = np.array([1, 1, 2, 2, 3, 3, 3, 4, 4], np.int32)
    pages = np.array([2, 7, 2, 7, 2, 5, 7, 2, 7], np.int32)
    vals = np.array([10, 70, 11, 71, 12, 50, 72, 13, 73], np.float32)
    f = {"seq": seq, "page_index": pages, "cols": {"hamming": vals}, "z": None,
         "channels": ["hamming"], "n_pairs": 4, "n_pages": 10, "block": None}

    M, kept = stages.dense_page_matrix(f, "active")
    assert kept.tolist() == [2, 5, 7] and M.shape == (4, 3)
    assert M[0].tolist() == [10, 0, 70] and M[2].tolist() == [12, 50, 72]
    Ma, ka = stages.dense_page_matrix(f, "all")
    assert Ma.shape == (4, 10) and ka.tolist() == list(range(10))
    assert Ma[:, 2].tolist() == [10, 11, 12, 13] and not Ma[:, 0].any()
    assert np.array_equal(Ma[:, kept], M)          # active is exactly the changing columns of all

    t = stages.window(f, 2, 1)
    assert t["page_axis"] and t["X"].shape == (3, 2, 3) and t["pages"].tolist() == [2, 5, 7]
    assert np.array_equal(t["X"][0], M[0:2])
    # every lens runs per page and reports the median across pages
    r = stages.run_lens(t, lambda x: stages.stats(x, ["mean", "max"]))
    assert r["names"] == ["mean_median", "max_median"] and r["rows"].shape == (3, 2)
    assert np.isclose(r["rows"][0, 0], np.median(t["X"][0].mean(axis=0)))
    assert np.isclose(r["rows"][0, 1], np.median(t["X"][0].max(axis=0)))
    # a collapsed tile is untouched by the wrapper
    flat = stages.window({"values": np.arange(8, dtype=np.float32), "channels": ["h"], "complex": False,
                          "block": None, "n_pages": 1}, 4, 4)
    assert stages.run_lens(flat, lambda x: stages.stats(x, ["mean"]))["names"] == ["mean"]

    # guards: budget, channel count, unknown mode
    for kw, word in (({"page_mode": "all", "max_bytes": 10}, "budget"), ({"page_mode": "nope"}, "unknown page mode")):
        try:
            stages.dense_page_matrix(f, **kw)
            assert False, kw
        except ValueError as e:
            assert word in str(e)
    try:
        stages.dense_page_matrix(dict(f, channels=["hamming", "l1"], cols={"hamming": vals, "l1": vals}), "active")
        assert False
    except ValueError as e:
        assert "one channel or a complex field" in str(e)

    # a complex field keeps its phase, and PLV reads the page axis directly
    rng = np.random.default_rng(0)
    T, P = 64, 40
    ang = np.zeros((T, P))
    ang[:, :P // 2] = np.linspace(0, 2 * np.pi, T)[:, None]
    ang[:, P // 2:] = rng.uniform(0, 2 * np.pi, (T, P // 2))
    fz = {"seq": np.repeat(np.arange(1, T + 1), P).astype(np.int32),
          "page_index": np.tile(np.arange(P), T).astype(np.int32), "cols": None,
          "z": (0.05 * np.exp(1j * ang)).astype(np.complex64).reshape(-1),
          "channels": ["hamming", "cosine"], "n_pairs": T, "n_pages": P, "block": None}
    tz = stages.window(fz, 32, 16)
    assert tz["complex"] and tz["X"].shape == (3, 32, P) and tz["X"].dtype == np.complex64
    ref = stages.baseline(tz, "cell", "rec0")
    assert len(ref["baseline_plv"]) == P                      # one PLV per page, not one per tile
    assert ref["baseline_plv"][:P // 2].mean() > ref["baseline_plv"][P // 2:].mean()
    out = stages.plv(tz, ref, 0.2, 0.7)
    got = dict(zip(out["names"], out["rows"][0]))
    assert got["plv_num_very_weak_stability"] + got["plv_num_moderate_stability"] == P


def test_lenses_find_a_known_period():
    """A period-8 series at W=32: the spectral peak is bin 4 and the quefrency peak is 8.

    Both were wrong before: the FFT's peak sat on bin 1 (a taper leaks DC into the low bins
    unless the mean is removed) and the cepstrum's sat on 31 (the taper's envelope enters
    the log). These are the regressions that pin the fixes.
    """
    rng = np.random.default_rng(0)
    T, W = 128, 32
    v = (0.05 + 0.02 * np.sin(2 * np.pi * np.arange(T) / 8) + 0.002 * rng.standard_normal(T)).astype(np.float32)
    for taper in ("rectangular", "hann"):
        tiles = stages.window({"values": v, "channels": ["h"], "complex": False, "block": None, "n_pages": 1}, W, 16, taper=taper)
        peak = stages.fft(tiles, "peak")
        assert peak["names"][0] == "fft_peak_bin"
        assert set(peak["rows"][:, 0].astype(int)) == {4}, (taper, peak["rows"][:, 0])
        assert set(stages.fft(tiles, "peak", detrend="none")["rows"][:, 0].astype(int)) == ({4} if taper == "rectangular" else {1})
        ceps = stages.cepstrum(tiles)
        assert ceps["names"] == ["cepstral_peak_idx", "ceps_peak_snr_db"]
        # the per-tile estimate wobbles on a noisy 32-sample window; the mode is the period
        peaks = ceps["rows"][:, 0].astype(int)
        assert np.bincount(peaks).argmax() == 8, (taper, peaks)
        assert np.all(ceps["rows"][:, 1] > 0)
        # the old path applied the window's taper; under Hann that moved the peak off 8
        tapered = np.array([stages._ceps_peak(t * stages._taper(W, taper))[0] for t in tiles["X"]], dtype=int)
        if taper == "hann":
            assert np.bincount(tapered).argmax() != 8, tapered
    # a flat series has no period to find: its cepstral SNR is lower than the periodic one's
    flat = (0.05 + 0.002 * rng.standard_normal(T)).astype(np.float32)
    tf = stages.window({"values": flat, "channels": ["h"], "complex": False, "block": None, "n_pages": 1}, W, 16)
    tp = stages.window({"values": v, "channels": ["h"], "complex": False, "block": None, "n_pages": 1}, W, 16)
    assert stages.cepstrum(tp)["rows"][:, 1].mean() > stages.cepstrum(tf)["rows"][:, 1].mean()


def test_deep_and_plv_discriminate():
    """deep and PLV must separate signals that differ; a lens returning a constant is not working."""
    rng = np.random.default_rng(1)
    T, W = 128, 32
    mk = lambda x, c=False: stages.window({"values": x, "channels": ["h"], "complex": c, "block": None, "n_pages": 1}, W, 16)
    periodic = (0.05 + 0.02 * np.sin(2 * np.pi * np.arange(T) / 8) + 0.002 * rng.standard_normal(T)).astype(np.float32)
    ramp = np.linspace(0.01, 0.10, T).astype(np.float32)
    feats = ["tau", "skew", "kurtosis", "entropy", "stat_pass_frac"]
    dp, dr = stages.deep(mk(periodic), feats), stages.deep(mk(ramp), feats)
    col = lambda blk, n: blk["rows"][:, blk["names"].index(n)]
    assert col(dr, "tau").mean() > col(dp, "tau").mean()          # a ramp decorrelates slowly
    assert col(dp, "stat_pass_frac").mean() > col(dr, "stat_pass_frac").mean()   # a ramp is not stationary
    assert abs(col(dp, "kurtosis").mean() + 1.4) < 0.3            # a sinusoid's excess kurtosis is about -1.5
    assert np.isclose(col(dr, "entropy").mean(), 4.0, atol=0.01)  # a ramp is uniform over 16 bins: log2(16)
    # PLV: a coherent phasor scores high against its own baseline, a random-phase one does not
    z0 = (0.05 * np.exp(1j * np.linspace(0, 2 * np.pi, T))).astype(np.complex64)
    z1 = (0.05 * np.exp(1j * rng.uniform(0, 2 * np.pi, T))).astype(np.complex64)
    ref = stages.baseline(mk(z0, True), "cell", "rec0")
    hi = stages.plv(mk(z0, True), ref, 0.2, 0.7)
    lo = stages.plv(mk(z1, True), ref, 0.2, 0.7)
    g = lambda blk, n: blk["rows"][:, blk["names"].index(n)].sum()
    assert g(hi, "plv_num_high_stability") > g(lo, "plv_num_high_stability")
    assert g(lo, "plv_num_very_weak_stability") > g(hi, "plv_num_very_weak_stability")


def test_wavelet_when_pywt_is_present():
    """Energy per level, Parseval-exact, with a level ceiling set by the filter and not by log2(W)."""
    try:
        import pywt  # noqa: F401
    except ImportError:
        print("skip: pywt not installed")
        return
    rng = np.random.default_rng(2)
    T, W = 128, 32
    mk = lambda x: stages.window({"values": x, "channels": ["h"], "complex": False, "block": None, "n_pages": 1}, W, 16)
    flat = (0.05 + 0.002 * rng.standard_normal(T)).astype(np.float32)
    burst = flat.copy()
    burst[40:48] += 0.05
    out = stages.wavelet(mk(burst), "db4", 2)
    assert out["names"] == ["wav_l0_energy", "wav_l1_energy", "wav_l2_energy"]
    assert out["rows"].shape == (7, 3) and not np.isnan(out["rows"]).any()
    # the burst lands in the tiles that contain it, and nowhere else
    e_flat, e_burst = stages.wavelet(mk(flat), "db4", 2)["rows"].sum(axis=1), out["rows"].sum(axis=1)
    hit = e_burst > e_flat * 1.2
    assert hit.sum() >= 1 and not hit[0] and not hit[-1], (e_flat, e_burst)
    # periodization is energy-preserving: coefficient energy equals the tile's
    tiles = mk(burst)
    for t in range(tiles["X"].shape[0]):
        assert np.isclose(stages.wavelet({**tiles, "X": tiles["X"][t:t + 1], "keys": tiles["keys"][t:t + 1]}, "db4", 2)["rows"].sum(),
                          float(np.square(tiles["X"][t].astype(np.float64)).sum()), rtol=1e-5)
    # the ceiling is the filter length: at W=32 haar takes 5, db4 two, sym5 one, coif3 none
    assert [stages.wavelet_max_level(f, 32) for f in ("haar", "db4", "sym5", "coif3")] == [5, 2, 1, 0]
    for fam, lv in (("db4", 3), ("coif3", 1), ("db4", 0)):
        try:
            stages.wavelet(mk(burst), fam, lv)
            assert False, (fam, lv)
        except ValueError as e:
            assert "level" in str(e)
    # a continuous family is refused by name rather than raising out of pywt
    try:
        stages.wavelet(mk(burst), "morl", 1)
        assert False
    except stages.NotImplementedStage as e:
        assert "discrete" in str(e)
    # the console offers only families wavedec accepts
    from plan10_analysis.modules import build_modules
    wav = build_modules()["feature_sources"]["wavelet"]
    assert "morl" not in wav["families"] and "db4" in wav["families"]
    assert "morl" in wav["excluded_continuous"]
    assert wav["filter_len"]["db4"] == 8 and wav["filter_len"]["haar"] == 2


def test_scattering_when_kymatio_is_present():
    """Translation invariance is the property this lens is here for; pin it and the J ceiling."""
    try:
        import kymatio  # noqa: F401
    except ImportError:
        print("skip: kymatio not installed")
        return
    W = 64
    rng = np.random.default_rng(3)
    mk = lambda x: {"X": np.asarray(x, np.float32).reshape(1, -1), "keys": [(None, 0, 1)], "w": W, "h": W,
                    "taper": "rectangular", "channels": ["h"], "complex": False, "series_mean": 0.0, "series_std": 1.0}

    def burst(pos, width=8, amp=0.05):
        v = 0.05 + 0.002 * rng.standard_normal(W)
        v[pos:pos + width] += amp
        return v.astype(np.float32)

    J = stages.scattering_max_J(W, 4)
    assert J == 2, J                      # measured from kymatio, not assumed
    assert stages.scattering_max_J(16, 4) == 0    # too short for any J
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(a))
    a, b = burst(8), burst(40)
    sa, sb = (stages.scattering(mk(x), J, 4) for x in (a, b))
    assert sa["names"] == sb["names"] and len(sa["names"]) == sa["rows"].shape[1]
    assert all(n.startswith("scat_o") for n in sa["names"])
    assert np.isfinite(sa["rows"]).all()
    shift = rel(sa["rows"][0], sb["rows"][0])
    # invariant to where the burst sits, and far more so than a spectrum of the same tile
    assert shift < 0.02, shift
    assert shift < rel(stages.fft(mk(a), "bands")["rows"][0], stages.fft(mk(b), "bands")["rows"][0])
    # still separates different shapes by much more than it moves under a shift
    flat = (0.05 + 0.002 * rng.standard_normal(W)).astype(np.float32)
    per = (0.05 + 0.02 * np.sin(2 * np.pi * np.arange(W) / 8)).astype(np.float32)
    assert rel(stages.scattering(mk(flat), J, 4)["rows"][0], stages.scattering(mk(per), J, 4)["rows"][0]) > 5 * shift
    for bad in (J + 1, 0):
        try:
            stages.scattering(mk(a), bad, 4)
            assert False, bad
        except ValueError as e:
            assert "allows 1 to" in str(e)
    # the ceiling the console enforces without kymatio matches the measured one
    grid = build_modules()["feature_sources"]["scattering"]
    assert grid["available"] and grid["max_J"]["64"]["4"] == J and grid["max_J"]["16"]["4"] == 0


def test_msc_welch_against_the_degenerate_legacy():
    """Coherence must be 1 for a repeating spectrum and near 1/n_pairs for independent segments."""
    W, iw, ih = 512, 64, 64
    t = np.arange(W)
    rng = np.random.default_rng(0)
    mk = lambda x: {"X": np.asarray(x, np.float32).reshape(1, W), "keys": [(None, 0, 1)], "w": W, "h": W,
                    "taper": "rectangular", "channels": ["h"], "complex": False, "series_mean": 0.0, "series_std": 1.0}
    sine = 0.05 + 0.02 * np.sin(2 * np.pi * t / 16)
    chirp = 0.05 + 0.02 * np.sin(2 * np.pi * t ** 2 / (10 * W))
    noise = 0.05 + 0.02 * rng.standard_normal(W)
    g = lambda r, k: float(r["rows"][0][r["names"].index(k)])

    r_sine, r_chirp, r_noise = (stages.msc(mk(v), iw, ih) for v in (sine, chirp, noise))
    assert r_sine["names"] == ["msc_peak_snr_db", "msc_mean", "msc_max", "msc_weighted"]
    # a spectrum that repeats exactly is fully coherent; a drifting one and noise are not
    assert g(r_sine, "msc_weighted") > 0.99
    assert g(r_chirp, "msc_weighted") < 0.4 and g(r_noise, "msc_weighted") < 0.4
    # independent segments sit near 1/n_pairs, the bias of the estimator at 7 pairs
    assert 0.05 < g(r_noise, "msc_weighted") < 0.35, g(r_noise, "msc_weighted")
    for r in (r_sine, r_chirp, r_noise):
        for k in ("msc_mean", "msc_max", "msc_weighted"):
            assert 0.0 <= g(r, k) <= 1.0 + 1e-9, (k, g(r, k))
    # without detrending, DC owns the coherence and nothing separates
    assert min(g(stages.msc(mk(v), iw, ih, detrend="none"), "msc_weighted") for v in (sine, chirp, noise)) > 0.9

    # the legacy path is identically 1 wherever power exists: two INDEPENDENT windows score 1
    from coherence_temp_spec_stability.magnitude_squared_coherence import MagnitudeSquaredCoherence
    op = MagnitudeSquaredCoherence(window_size=64, window_step=32)
    c = op.compute_pair_msc(rng.standard_normal((64, 1)), rng.standard_normal((64, 1)))[:, 0]
    assert c.min() > 0.999, c.min()
    # and its msc_mean equals the fraction of bins holding power, which is occupancy
    legacy = stages.msc(mk(sine), iw, ih, method="legacy_adjacent")
    P = np.abs(np.fft.rfft(sine[:iw])) ** 2
    assert abs(g(legacy, "msc_mean") - float((P > P.max() * 1e-9).mean())) < 0.01

    # one channel is enough (it is not a channel pair), and several are independent
    two = np.stack([sine, noise], axis=1)
    tiles2 = mk(sine)
    tiles2["X"] = two.astype(np.float32).reshape(1, W, 2)
    tiles2["channels"] = ["hamming", "cosine"]
    r2 = stages.msc(tiles2, iw, ih)
    assert r2["names"][0].endswith(":hamming") and r2["names"][4].endswith(":cosine")
    assert np.allclose(r2["rows"][0][:4], r_sine["rows"][0], rtol=1e-5)
    assert np.allclose(r2["rows"][0][4:], r_noise["rows"][0], rtol=1e-5)

    # length rule differs per method: welch needs three segments, legacy two
    assert stages.msc_min_window(128, 64, "welch") == 256 and stages.msc_min_window(128, 64, "legacy_adjacent") == 192
    short = mk(sine)
    short["w"], short["X"] = 127, short["X"][:, :127]
    for meth in ("welch", "legacy_adjacent"):
        try:
            stages.msc(short, iw, ih, method=meth)
            assert False, meth
        except ValueError as e:
            assert "at least" in str(e)


def test_benign_envelope_and_deviation():
    """A p5-p95 band must leave about 10% of benign values outside per feature, and more as data shifts."""
    rng = np.random.default_rng(0)
    names = ["mean", "std", "cov", "max"]
    mkf = lambda X: {"names": names, "rows": np.asarray(X, np.float32), "keys": [(None, i, i) for i in range(len(X))]}
    benign = [(f"r{i}", mkf(rng.normal(1.0, 0.1, size=(40, 4)))) for i in range(3)]
    env = stages.baseline_envelope(benign)
    assert env["kind"] == "envelope" and env["names"] == names and env["n_rows"] == 120
    assert env["recordings"] == ["r0", "r1", "r2"] and env["q"] == [5.0, 95.0]
    assert np.all(env["lo"] < env["hi"]) and np.all(env["width"] > 0)

    col = lambda d, k: d["rows"][:, d["names"].index(k)]
    out_by = lambda X: float(col(stages.deviation(mkf(X), env), "dev_n_outside").mean())
    same = out_by(rng.normal(1.0, 0.1, size=(400, 4)))
    assert 0.2 < same < 0.7, same                      # about 10% of 4 features
    # the further from normal, the more features leave the band
    assert same < out_by(rng.normal(1.1, 0.1, size=(400, 4))) < out_by(rng.normal(1.5, 0.1, size=(400, 4)))
    assert out_by(rng.normal(1.5, 0.1, size=(400, 4))) > 3.5
    d = stages.deviation(mkf(rng.normal(1.5, 0.1, size=(50, 4))), env)
    assert d["names"] == ["dev_n_outside", "dev_frac_outside", "dev_max_excess", "dev_total_excess"]
    assert np.all(col(d, "dev_frac_outside") <= 1.0) and np.all(col(d, "dev_max_excess") >= 0)

    # NaN counts as inside, as normal_profile.py does
    X = np.full((2, 4), 1.0)
    X[0, 0] = np.nan
    X[1, 1] = 99.0
    dn = stages.deviation(mkf(X), env)
    assert col(dn, "dev_n_outside").tolist() == [0.0, 1.0]

    # a zero-width band still yields an excess: the scale falls back to |median|, then 1.0.
    # Without it a feature the benign set pins to one value could be violated by any margin
    # and still report 0, which is how this first behaved end to end.
    flat = stages.baseline_envelope([("r", {"names": ["a", "b"], "rows": np.array([[1.0, 5.0]] * 3, np.float32), "keys": []})])
    assert np.all(flat["width"] == 0)
    dz = stages.deviation({"names": ["a", "b"], "rows": np.array([[1.0, 5.0], [2.0, 5.0], [1.0, 0.0]], np.float32), "keys": []}, flat)
    assert col(dz, "dev_n_outside").tolist() == [0.0, 1.0, 1.0]
    assert np.allclose(col(dz, "dev_max_excess"), [0.0, 1.0, 1.0])       # 1.0/|1.0| and 5.0/|5.0|
    zero_med = stages.baseline_envelope([("r", {"names": ["a"], "rows": np.zeros((3, 1), np.float32), "keys": []})])
    dz0 = stages.deviation({"names": ["a"], "rows": np.array([[3.0]], np.float32), "keys": []}, zero_med)
    assert np.isclose(col(dz0, "dev_max_excess")[0], 3.0)                # scale falls through to 1.0

    # a percentile pair that is not a band, and the two reference kinds are not interchangeable
    for lo, hi in ((95.0, 5.0), (5.0, 5.0)):
        e = stages.baseline_envelope(benign, lo, hi)
        assert np.all(e["width"] <= 0) or lo >= hi          # scheme.py refuses this before the runner sees it
    try:
        stages.deviation(mkf(np.zeros((5, 4))), {"kind": "plv"})
        assert False
    except ValueError as e:
        assert "envelope reference" in str(e)
    try:
        stages.baseline({"complex": True, "X": np.zeros((2, 4), np.complex64), "w": 4, "channels": ["a"]}, "benign", "r")
        assert False
    except ValueError as e:
        assert "reads features" in str(e)
    # shape mismatches are named, not left to numpy
    for rows, why in ((np.zeros((5, 3)), "columns"), (np.zeros((5, 5)), "columns")):
        try:
            stages.deviation({"names": names, "rows": rows, "keys": []}, env)
            assert False
        except ValueError as e:
            assert why in str(e)
    try:
        stages.baseline_envelope([("a", mkf(np.zeros((5, 4)))), ("b", {"names": ["x"], "rows": np.zeros((5, 1)), "keys": []})])
        assert False
    except ValueError as e:
        assert "different feature names" in str(e)
    try:
        stages.baseline_envelope([("a", mkf(np.zeros((1, 4))))])
        assert False
    except ValueError as e:
        assert "at least 2 rows" in str(e)


def test_unimplemented_modules_refuse_by_name():
    tiles = stages.window({"values": np.arange(32, dtype=np.float32), "channels": ["h"], "complex": False, "block": None, "n_pages": 1}, 16, 8)
    try:
        import kymatio  # noqa: F401
    except ImportError:
        try:
            stages.scattering(tiles, 2, 4)
            assert False
        except stages.NotImplementedStage as e:
            assert "kymatio" in str(e)
    try:
        import pywt  # noqa: F401
    except ImportError:
        try:
            stages.wavelet(tiles, "db4", 2)
            assert False
        except stages.NotImplementedStage as e:
            assert "pywt" in str(e)


def test_b1_example_end_to_end_and_apf_equals_fixture():
    if not _have_tools():
        return
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        root, manifest, mp, changes = _corpus(td, n_snapshots=10)
        s = _example(manifest, "b1")
        win = next(n for n in s["nodes"] if n["module"] == "window")
        win["params"].update(w=4, h=2)
        sp = td / "b1.json"
        sp.write_text(json.dumps(s))
        rc = executor.run(sp, td / "out", {"kind": "local", "root": str(root)}, td / "l1", speed=2, manifest_path=mp)
        assert rc == 0, (td / "out" / s["label"] / "run.log").read_text()
        out = td / "out" / s["label"]
        st = json.loads((out / "status.json").read_text())
        assert st["state"] == "done" and st["n_recordings"] == 3
        z = np.load(out / "features.npz")
        names = z["feature_names"].tolist()
        assert names == ["mean", "std", "cov", "median", "max", "p95", "peak2med", "duty"]
        n_windows_per_rec = (9 - 4) // 2 + 1
        assert z["X"].shape == (3 * n_windows_per_rec, 8)
        keys = z["tile_keys"]
        assert set(keys["workload"].tolist()) == {"mem_synth_a_v2", "cpu_synth_b_v2"}
        # APF from the fixture: 7 changed pages per pair -> 7/1024, constant, so mean == max == 7/1024
        assert np.allclose(z["X"][:, 0], 7 / 1024) and np.allclose(z["X"][:, 4], 7 / 1024)
        side = json.loads((out / "sidecar.json").read_text())
        assert side["acknowledged"][0]["id"] == "no_substrate" and side["speed"] == 2 and side["n_rows"] == z["X"].shape[0]
        assert side["source"]["kind"] == "local" and side["differ"]["path"].endswith("live_delta_calc_modular")
        assert (out / "features.csv").read_text().splitlines()[0].startswith("recording,workload,family,block,t_index,seq_start,mean")


def test_complex_and_plv_examples_end_to_end():
    if not _have_tools():
        return
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        root, manifest, mp, _ = _corpus(td, n_snapshots=12)
        # complex -> fft + cepstrum -> concat
        s = _example(manifest, "complex")
        next(n for n in s["nodes"] if n["module"] == "window")["params"].update(w=8, h=4)
        sp = td / "cx.json"
        sp.write_text(json.dumps(s))
        rc = executor.run(sp, td / "out", {"kind": "local", "root": str(root)}, td / "l1", speed=2, manifest_path=mp)
        assert rc == 0, (td / "out" / s["label"] / "run.log").read_text()
        z = np.load(td / "out" / s["label"] / "features.npz")
        names = z["feature_names"].tolist()
        assert names == ["fft_band0", "fft_band1", "fft_band2", "fft_band3", "cepstral_peak_idx", "ceps_peak_snr_db"], names
        assert z["X"].shape[0] == 3 * ((11 - 8) // 4 + 1)
        # plv with a baseline fitted on the first recording (L1 store reused from the run above)
        s = _example(manifest, "plv")
        next(n for n in s["nodes"] if n["module"] == "window")["params"].update(w=8, h=4)
        sp = td / "plv.json"
        sp.write_text(json.dumps(s))
        rc = executor.run(sp, td / "out", {"kind": "local", "root": str(root)}, td / "l1", speed=2, manifest_path=mp)
        assert rc == 0, (td / "out" / s["label"] / "run.log").read_text()
        z = np.load(td / "out" / s["label"] / "features.npz")
        assert all(n.startswith("plv_num_") for n in z["feature_names"].tolist()) and z["X"].shape[0] == 3
        log = (td / "out" / s["label"] / "run.log").read_text()
        assert "L1 store reused" in log and "baseline fitted on" in log


def test_refusal_and_stop():
    if not _have_tools():
        return
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        root, manifest, mp, _ = _corpus(td, n_snapshots=6)
        s = _example(manifest, "b1")
        next(n for n in s["nodes"] if n["module"] == "window")["params"].update(w=4, h=2)
        s["acknowledged"] = []                      # unacknowledged warning -> refused, exit 2
        sp = td / "b1.json"
        sp.write_text(json.dumps(s))
        rc = executor.run(sp, td / "out", {"kind": "local", "root": str(root)}, td / "l1", speed=2, manifest_path=mp)
        assert rc == 2
        st = json.loads((td / "out" / s["label"] / "status.json").read_text())
        assert st["state"] == "refused"
        # --acknowledge-all lets it through; a stop written while extracting is honoured
        out = td / "out2" / s["label"]
        out.mkdir(parents=True)
        (out / "control.json").write_text(json.dumps({"command": "stop"}))
        rc = executor.run(sp, td / "out2", {"kind": "local", "root": str(root)}, td / "l1b", speed=2, manifest_path=mp, acknowledge_all=True)
        assert rc == 130 and json.loads((out / "status.json").read_text())["state"] == "stopped"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
