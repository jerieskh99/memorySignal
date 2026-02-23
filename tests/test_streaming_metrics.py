from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coherence_temp_spec_stability"))

from streaming_metrics import (  # noqa: E402
    CepstrumFeatureAccumulator,
    PLVAccumulator,
    SlidingWindowBuffer,
    WelchMSCAccumulator,
    offline_plv,
    offline_plv_from_complex,
    offline_welch_msc,
    offline_windowed_cepstrum_features,
    run_streaming_on_time_series,
)


def _stream_welch_online(
    x: np.ndarray,
    y: np.ndarray,
    *,
    win_len: int,
    hop_len: int,
    nfft: int,
    window: str = "hann",
    eps: float = 1e-10,
    detrend: bool = False,
    onesided: bool = True,
) -> np.ndarray:
    xbuf = SlidingWindowBuffer(win_len, hop_len)
    ybuf = SlidingWindowBuffer(win_len, hop_len)
    acc = WelchMSCAccumulator(
        nfft=nfft,
        window=window,
        eps=eps,
        detrend=detrend,
        onesided=onesided,
    )

    x2 = x if x.ndim == 2 else x[:, None]
    y2 = y if y.ndim == 2 else y[:, None]

    for t in range(x2.shape[0]):
        xwins = xbuf.push(x2[t])
        ywins = ybuf.push(y2[t])
        assert len(xwins) == len(ywins)
        for xw, yw in zip(xwins, ywins):
            acc.update_window(xw, yw)

    return acc.finalize()


def _offline_adjacent_window_msc(
    mag_trace: np.ndarray,
    *,
    win_len: int,
    hop_len: int,
    nfft: int,
    window: str,
    eps: float,
    detrend: bool,
) -> np.ndarray:
    x = mag_trace if mag_trace.ndim == 2 else mag_trace[:, None]
    starts = list(range(0, x.shape[0] - win_len + 1, hop_len))
    if len(starts) < 2:
        raise ValueError("Need at least two windows for adjacent-window MSC")

    acc = WelchMSCAccumulator(
        nfft=nfft,
        window=window,
        eps=eps,
        detrend=detrend,
        onesided=True,
    )
    for i in range(1, len(starts)):
        s0 = starts[i - 1]
        s1 = starts[i]
        acc.update_window(x[s0 : s0 + win_len], x[s1 : s1 + win_len])
    return acc.finalize()


def test_offline_online_welch_msc_equivalence_with_overlap_and_complex():
    rng = np.random.default_rng(123)
    t = 2048

    base = np.exp(1j * (0.035 * np.arange(t)))
    x = base + 0.15 * (rng.standard_normal(t) + 1j * rng.standard_normal(t))
    y = 0.7 * np.roll(base, 3) + 0.15 * (rng.standard_normal(t) + 1j * rng.standard_normal(t))

    win_len = 128
    hop_len = 64
    nfft = 256

    offline = offline_welch_msc(
        x,
        y,
        win_len,
        hop_len,
        nfft,
        window="hann",
        eps=1e-10,
        detrend=True,
        onesided=False,
    )
    online = _stream_welch_online(
        x,
        y,
        win_len=win_len,
        hop_len=hop_len,
        nfft=nfft,
        window="hann",
        eps=1e-10,
        detrend=True,
        onesided=False,
    )

    assert np.allclose(offline, online, rtol=1e-12, atol=1e-12)


def test_offline_online_plv_equivalence_scalar_and_complex_pair():
    rng = np.random.default_rng(7)
    t = 4096

    phase_diff = 0.4 * np.sin(0.02 * np.arange(t)) + 0.05 * rng.standard_normal(t)
    offline_scalar = offline_plv(phase_diff)

    acc = PLVAccumulator()
    for chunk in np.array_split(phase_diff, 17):
        acc.update_phase_diff(chunk)
    online_scalar = acc.finalize()
    assert np.allclose(offline_scalar, online_scalar, rtol=1e-13, atol=1e-13)

    zx = np.exp(1j * (0.01 * np.arange(t) + 0.2 * rng.standard_normal(t)))
    zy = np.exp(1j * (0.01 * np.arange(t) + phase_diff))
    offline_complex = offline_plv_from_complex(zx, zy)

    acc2 = PLVAccumulator()
    for sl in np.array_split(np.arange(t), 29):
        dp = np.angle(zx[sl]) - np.angle(zy[sl])
        acc2.update_phase_diff(dp)
    online_complex = acc2.finalize()

    assert np.allclose(offline_complex, online_complex, rtol=1e-13, atol=1e-13)


def test_offline_online_windowed_cepstrum_features_equivalence():
    rng = np.random.default_rng(202)
    t = 3000
    n = 3

    x = np.zeros((t, n), dtype=float)
    for i in range(n):
        sig = np.sin(2 * np.pi * (0.01 + 0.003 * i) * np.arange(t))
        x[:, i] = sig + 0.2 * rng.standard_normal(t)

    win_len = 192
    hop_len = 96
    nfft = 256

    offline = offline_windowed_cepstrum_features(
        x,
        win_len,
        hop_len,
        nfft,
        window="hamming",
        eps=1e-10,
        detrend=False,
        min_quef_idx=4,
    )

    buf = SlidingWindowBuffer(win_len, hop_len)
    online_acc = CepstrumFeatureAccumulator(
        nfft=nfft,
        window="hamming",
        eps=1e-10,
        detrend=False,
        min_quef_idx=4,
    )

    for t_idx in range(t):
        for w in buf.push(x[t_idx]):
            online_acc.update_window(w)
    online = online_acc.finalize()

    keys = [
        "cepstral_peak_idx_mean",
        "cepstral_peak_idx_var",
        "cepstral_peak_mag_mean",
        "cepstral_peak_mag_var",
        "cepstral_band_energy_mean",
        "cepstral_band_energy_var",
        "cepstral_peak_idx_median",
    ]
    for k in keys:
        assert np.allclose(offline[k], online[k], rtol=1e-12, atol=1e-12)


def test_streaming_run_processor_matches_offline_windowed_definitions():
    rng = np.random.default_rng(99)
    t = 1500
    n = 8
    delta = (
        np.exp(1j * (0.015 * np.arange(t)[:, None] + 0.12 * rng.standard_normal((t, n))))
        * (1.0 + 0.25 * rng.standard_normal((t, n)))
    )

    win_len = 128
    hop_len = 64
    nfft = 128

    online = run_streaming_on_time_series(
        delta,
        win_len=win_len,
        hop_len=hop_len,
        nfft=nfft,
        window="hann",
        eps=1e-10,
        detrend=True,
        min_quef_idx=3,
    )

    offline_plv_per_page = offline_plv(np.angle(delta))
    offline_cep = offline_windowed_cepstrum_features(
        np.abs(delta),
        win_len,
        hop_len,
        nfft,
        window="hann",
        eps=1e-10,
        detrend=True,
        min_quef_idx=3,
    )
    offline_adj_msc = _offline_adjacent_window_msc(
        np.abs(delta),
        win_len=win_len,
        hop_len=hop_len,
        nfft=nfft,
        window="hann",
        eps=1e-10,
        detrend=True,
    )

    assert np.allclose(online["plv"]["plv_per_page"], offline_plv_per_page, rtol=1e-12, atol=1e-12)
    assert np.allclose(online["msc"]["msc_spectrum"], offline_adj_msc, rtol=1e-12, atol=1e-12)
    assert np.allclose(
        online["cepstrum"]["cepstral_peak_idx_median"],
        offline_cep["cepstral_peak_idx_median"],
        rtol=1e-12,
        atol=1e-12,
    )
