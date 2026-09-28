"""nulls.py: surrogates, unit-level shuffles, the null summary (SPEC 3.2; CR 2.1 items 3, 9)."""
import numpy as np

from _b2_common import S
from plan11_encoding_ladder import nulls as NL


def test_phase_randomize_preserves_amplitude_spectrum_and_mean():
    x = np.random.default_rng(1).random(120) + np.sin(np.arange(120) / 4.0)
    sur = NL.surrogates(x, 5)
    assert sur.shape == (5, 120)
    for s in sur:
        assert np.allclose(np.abs(np.fft.rfft(x - x.mean())), np.abs(np.fft.rfft(s - s.mean())), atol=1e-9)
        assert abs(s.mean() - x.mean()) < 1e-9
        assert not np.allclose(s, x)
    assert np.allclose(NL.surrogates(x, 3, seed=7), NL.surrogates(x, 3, seed=7))


def test_unit_level_shuffles_keep_the_structure():
    cells = np.array([f"c{i}" for i in range(24)])
    kern = np.array([f"k{i // 2}" for i in range(24)])
    arch = np.array(["A"] * 12 + ["B"] * 6 + ["C"] * 4 + ["D"] * 2)
    rng = np.random.default_rng(0)
    p = NL.shuffle_labels_units(cells, kern, arch, "loko", "archetype", rng)
    assert sorted(p.tolist()) == sorted(arch.tolist())
    for k in set(kern):
        assert len(set(p[kern == k].tolist())) == 1          # cells inherit one draw per kernel
    q = NL.shuffle_labels_units(cells, kern, arch, "loro", "kernel", rng)
    assert sorted(q.tolist()) == sorted(kern.tolist()) and not np.array_equal(q, kern)
    r = NL.shuffle_labels_units(cells, kern, np.array(["01c", "01c1"] * 12), "loko", "campaign", rng)
    assert sorted(r.tolist()) == ["01c"] * 12 + ["01c1"] * 12


def test_null_summary_strict_exceedance_and_rank():
    null = np.array([0.1, 0.2, 0.3, 0.9, 0.9])
    s = NL.null_summary(0.9, null)
    assert s["exceeds"] is False and s["rank"] == 3 and s["n"] == 5 and abs(s["spread"] - (s["p95"] - s["p05"])) < 1e-12
    assert NL.null_summary(0.95, null)["exceeds"] is True
    assert NL.null_summary(0.5, np.zeros(0))["p95"] is None
    x = np.arange(10.0)
    y = NL.order_shuffle(x, np.random.default_rng(3))
    assert sorted(y.tolist()) == x.tolist() and not np.array_equal(x, y)
