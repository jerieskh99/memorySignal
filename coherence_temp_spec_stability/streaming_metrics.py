import json
from dataclasses import dataclass
from typing import Any
from unittest import defaultTestLoader

import numpy as np


def _resolve_window(window: str | np.ndarray, win_len: int) -> np.ndarray:
    if isinstance(window, str):
        if window == "hann":
            return np.hanning(win_len)
        if window == "hamming":
            return np.hamming(win_len)
        if window in {"boxcar", "rect", "rectangular"}:
            return np.ones(win_len, dtype=float)
        raise ValueError(f"Unsupported window type: {window}")

    w = np.asarray(window, dtype=float)
    if w.ndim != 1 or w.shape[0] != win_len:
        raise ValueError(f"Window must be 1D with length {win_len}, got {w.shape}")
    return w


def _as_2d_time_major(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")


class SlidingWindowBuffer:
    """Streaming sliding window slicer with fixed hop."""

    def __init__(self, win_len: int, hop_len: int):
        if win_len <= 0:
            raise ValueError("win_len must be positive")
        if hop_len <= 0:
            raise ValueError("hop_len must be positive")

        self.win_len = int(win_len)
        self.hop_len = int(hop_len)
        self._buf: list[np.ndarray] = []
        self._buf_start_abs = 0
        self._next_start_abs = 0

    def push(self, sample: np.ndarray | float | complex) -> list[np.ndarray]:
        self._buf.append(np.asarray(sample))
        windows: list[np.ndarray] = []

        abs_end = self._buf_start_abs + len(self._buf)
        while self._next_start_abs + self.win_len <= abs_end:
            local_start = self._next_start_abs - self._buf_start_abs
            local_end = local_start + self.win_len
            windows.append(np.stack(self._buf[local_start:local_end], axis=0))
            self._next_start_abs += self.hop_len

        drop = self._next_start_abs - self._buf_start_abs
        if drop > 0:
            del self._buf[:drop]
            self._buf_start_abs = self._next_start_abs

        return windows


class WelchMSCAccumulator:
    """Online Welch MSC accumulator for 1D or [T, C] windows."""

    def __init__(
        self,
        nfft: int | None = None,
        *,
        window: str | np.ndarray = "hann",
        eps: float = 1e-10,
        detrend: bool = False,
        onesided: bool = True,
    ):
        self.nfft = nfft
        self.window = window
        self.eps = float(eps)
        self.detrend = bool(detrend)
        self.onesided = bool(onesided)

        self._win_len: int | None = None
        self._w: np.ndarray | None = None
        self._sxx: np.ndarray | None = None
        self._syy: np.ndarray | None = None
        self._sxy: np.ndarray | None = None
        self._num_windows = 0

    def _setup_for_length(self, win_len: int):
        if self._win_len is None:
            self._win_len = win_len
            self._w = _resolve_window(self.window, win_len)
            if self.nfft is None:
                self.nfft = win_len
            if self.nfft <= 0:
                raise ValueError("nfft must be positive")
        elif self._win_len != win_len:
            raise ValueError(f"Window length mismatch: expected {self._win_len}, got {win_len}")

    def update_window(self, x_win: np.ndarray, y_win: np.ndarray) -> None:
        x = _as_2d_time_major(x_win)
        y = _as_2d_time_major(y_win)

        if x.shape != y.shape:
            raise ValueError(f"x_win and y_win must match shape, got {x.shape} vs {y.shape}")

        self._setup_for_length(x.shape[0])
        assert self._w is not None
        assert self.nfft is not None

        if self.detrend:
            x = x - np.mean(x, axis=0, keepdims=True)
            y = y - np.mean(y, axis=0, keepdims=True)

        xw = x * self._w[:, None]
        yw = y * self._w[:, None]

        if self.onesided and np.isrealobj(xw) and np.isrealobj(yw):
            xf = np.fft.rfft(xw, n=self.nfft, axis=0)
            yf = np.fft.rfft(yw, n=self.nfft, axis=0)
        else:
            xf = np.fft.fft(xw, n=self.nfft, axis=0)
            yf = np.fft.fft(yw, n=self.nfft, axis=0)

        pxx = np.abs(xf) ** 2
        pyy = np.abs(yf) ** 2
        pxy = xf * np.conjugate(yf)

        if self._sxx is None:
            self._sxx = np.zeros_like(pxx, dtype=float)
            self._syy = np.zeros_like(pyy, dtype=float)
            self._sxy = np.zeros_like(pxy, dtype=np.complex128)

        self._sxx += pxx
        self._syy += pyy
        self._sxy += pxy
        self._num_windows += 1

    def get_spectra(self) -> dict[str, np.ndarray]:
        if self._num_windows == 0 or self._sxx is None or self._syy is None or self._sxy is None:
            raise ValueError("No windows accumulated")
        return {
            "Sxx": self._sxx.copy(),
            "Syy": self._syy.copy(),
            "Sxy": self._sxy.copy(),
            "num_windows": np.array(self._num_windows, dtype=int),
        }

    def finalize(self) -> np.ndarray:
        if self._num_windows == 0 or self._sxx is None or self._syy is None or self._sxy is None:
            raise ValueError("No windows accumulated")

        msc = (np.abs(self._sxy) ** 2) / (self._sxx * self._syy + self.eps)
        if msc.ndim == 2 and msc.shape[1] == 1:
            return msc[:, 0]
        return msc


class PLVAccumulator:
    """Online PLV accumulator using C = sum(exp(j*delta_phi))."""

    def __init__(self, num_channels: int | None = None):
        self.num_channels = num_channels
        self._complex_sum: np.ndarray | np.complex128 | None = None
        self._count = 0

    def update_phase_diff(self, delta_phi_array_or_scalar: np.ndarray | float) -> None:
        arr = np.asarray(delta_phi_array_or_scalar)

        if self.num_channels is None:
            vals = np.exp(1j * arr.ravel())
            if self._complex_sum is None:
                self._complex_sum = np.complex128(0.0 + 0.0j)
            self._complex_sum += np.sum(vals)
            self._count += vals.size
            return

        if arr.ndim == 1:
            if arr.shape[0] != self.num_channels:
                raise ValueError(f"Expected vector length {self.num_channels}, got {arr.shape[0]}")
            arr2 = arr[None, :]
        elif arr.ndim == 2:
            if arr.shape[1] != self.num_channels:
                raise ValueError(f"Expected second dim {self.num_channels}, got {arr.shape[1]}")
            arr2 = arr
        else:
            raise ValueError(f"Expected 1D or 2D array for channel PLV, got {arr.shape}")

        vals = np.exp(1j * arr2)
        if self._complex_sum is None:
            self._complex_sum = np.zeros(self.num_channels, dtype=np.complex128)
        self._complex_sum += np.sum(vals, axis=0)
        self._count += arr2.shape[0]

    def finalize(self) -> np.ndarray | float:
        if self._count == 0 or self._complex_sum is None:
            raise ValueError("No samples accumulated")

        plv = np.abs(self._complex_sum / self._count)
        if isinstance(plv, np.ndarray) and plv.ndim == 0:
            return float(plv)
        return plv


class CepstrumWindowProcessor:
    """Compute short-time (windowed) real cepstrum for each window."""

    def __init__(
        self,
        nfft: int,
        *,
        window: str | np.ndarray = "hann",
        eps: float = 1e-10,
        detrend: bool = False,
    ):
        if nfft <= 0:
            raise ValueError("nfft must be positive")
        self.nfft = int(nfft)
        self.window = window
        self.eps = float(eps)
        self.detrend = bool(detrend)
        self._win_len: int | None = None
        self._w: np.ndarray | None = None

    def _setup_for_length(self, win_len: int):
        if self._win_len is None:
            self._win_len = win_len
            self._w = _resolve_window(self.window, win_len)
        elif self._win_len != win_len:
            raise ValueError(f"Window length mismatch: expected {self._win_len}, got {win_len}")

    def process_window(self, x_win: np.ndarray) -> np.ndarray:
        x = _as_2d_time_major(x_win)
        self._setup_for_length(x.shape[0])
        assert self._w is not None

        if self.detrend:
            x = x - np.mean(x, axis=0, keepdims=True)

        xw = x * self._w[:, None]
        xf = np.fft.fft(xw, n=self.nfft, axis=0)
        log_power = np.log(np.abs(xf) ** 2 + self.eps)
        cep = np.real(np.fft.ifft(log_power, axis=0))

        if cep.shape[1] == 1:
            return cep[:, 0]
        return cep

    @staticmethod
    def extract_features(
        cepstrum: np.ndarray,
        *,
        min_quef_idx: int = 1,
        max_quef_idx: int | None = None,
    ) -> dict[str, np.ndarray]:
        c = _as_2d_time_major(cepstrum)
        q = c.shape[0]

        if max_quef_idx is None:
            max_quef_idx = q
        min_quef_idx = int(min_quef_idx)
        max_quef_idx = int(max_quef_idx)
        if not (0 <= min_quef_idx < max_quef_idx <= q):
            raise ValueError(
                f"Invalid quefrency band: min={min_quef_idx}, max={max_quef_idx}, q={q}"
            )

        c_abs = np.abs(c)
        band = c_abs[min_quef_idx:max_quef_idx, :]

        peak_rel = np.argmax(band, axis=0)
        peak_idx = peak_rel + min_quef_idx
        peak_mag = band[peak_rel, np.arange(band.shape[1])]

        features = {
            "cepstral_peak_idx": peak_idx.astype(int),
            "cepstral_peak_mag": peak_mag,
            "cepstral_band_energy": np.sum(band**2, axis=0),
            "cepstral_mean": np.mean(c, axis=0),
            "cepstral_var": np.var(c, axis=0),
        }

        if cepstrum.ndim == 1:
            return {k: v[0] if isinstance(v, np.ndarray) else v for k, v in features.items()}
        return features


class RunningStats:
    """Welford running mean/variance, scalar or vector."""

    def __init__(self):
        self.count = 0
        self.mean: np.ndarray | None = None
        self._m2: np.ndarray | None = None

    def update(self, x: np.ndarray | float) -> None:
        arr = np.asarray(x, dtype=float)

        if self.mean is None:
            self.mean = np.zeros_like(arr, dtype=float)
            self._m2 = np.zeros_like(arr, dtype=float)

        self.count += 1
        delta = arr - self.mean
        self.mean = self.mean + delta / self.count
        assert self._m2 is not None
        self._m2 = self._m2 + delta * (arr - self.mean)

    def finalize(self) -> dict[str, np.ndarray | float | int]:
        if self.count == 0 or self.mean is None or self._m2 is None:
            raise ValueError("No samples accumulated")

        var = self._m2 / self.count
        if self.mean.ndim == 0:
            return {
                "count": self.count,
                "mean": float(self.mean),
                "var": float(var),
            }

        return {
            "count": self.count,
            "mean": self.mean.copy(),
            "var": var.copy(),
        }


class CepstrumFeatureAccumulator:
    """Accumulate per-window cepstrum features and running stats."""

    def __init__(
        self,
        nfft: int,
        *,
        window: str | np.ndarray = "hann",
        eps: float = 1e-10,
        detrend: bool = False,
        min_quef_idx: int = 1,
        max_quef_idx: int | None = None,
        keep_per_window: bool = False,
    ):
        self.processor = CepstrumWindowProcessor(
            nfft=nfft,
            window=window,
            eps=eps,
            detrend=detrend,
        )
        self.min_quef_idx = min_quef_idx
        self.max_quef_idx = max_quef_idx
        self.keep_per_window = keep_per_window

        self._stats: dict[str, RunningStats] = {}
        self._peak_idx_windows: list[np.ndarray] = []
        self._per_window: list[dict[str, Any]] = []

    def update_window(self, x_win: np.ndarray) -> dict[str, np.ndarray]:
        cep = self.processor.process_window(x_win)
        feats = self.processor.extract_features(
            cep,
            min_quef_idx=self.min_quef_idx,
            max_quef_idx=self.max_quef_idx,
        )

        for key, value in feats.items():
            if key not in self._stats:
                self._stats[key] = RunningStats()
            self._stats[key].update(value)

        self._peak_idx_windows.append(np.asarray(feats["cepstral_peak_idx"]))
        if self.keep_per_window:
            self._per_window.append({k: np.asarray(v).copy() for k, v in feats.items()})

        return feats

    def finalize(self) -> dict[str, Any]:
        if not self._stats:
            raise ValueError("No windows accumulated")

        out: dict[str, Any] = {"num_windows": next(iter(self._stats.values())).count}
        for key, stat in self._stats.items():
            s = stat.finalize()
            out[f"{key}_mean"] = s["mean"]
            out[f"{key}_var"] = s["var"]

        peak_idx_arr = np.stack(self._peak_idx_windows, axis=0)
        out["cepstral_peak_idx_median"] = np.median(peak_idx_arr, axis=0)
        if self.keep_per_window:
            out["per_window"] = self._per_window

        return out


def offline_welch_msc(
    x: np.ndarray,
    y: np.ndarray,
    win_len: int,
    hop_len: int,
    nfft: int | None,
    *,
    window: str | np.ndarray = "hann",
    eps: float = 1e-10,
    detrend: bool = False,
    onesided: bool = True,
) -> np.ndarray:
    x2 = _as_2d_time_major(x)
    y2 = _as_2d_time_major(y)
    if x2.shape != y2.shape:
        raise ValueError(f"x and y must match shape, got {x2.shape} vs {y2.shape}")

    acc = WelchMSCAccumulator(
        nfft=nfft,
        window=window,
        eps=eps,
        detrend=detrend,
        onesided=onesided,
    )

    t = x2.shape[0]
    for start in range(0, t - win_len + 1, hop_len):
        end = start + win_len
        acc.update_window(x2[start:end, :], y2[start:end, :])

    return acc.finalize()


def offline_plv(delta_phi: np.ndarray) -> np.ndarray | float:
    arr = np.asarray(delta_phi)
    if arr.ndim == 1:
        return float(np.abs(np.mean(np.exp(1j * arr))))
    if arr.ndim == 2:
        return np.abs(np.mean(np.exp(1j * arr), axis=0))
    raise ValueError(f"Expected 1D or 2D phase array, got {arr.shape}")


def offline_plv_from_complex(zx: np.ndarray, zy: np.ndarray) -> np.ndarray | float:
    x = np.asarray(zx)
    y = np.asarray(zy)
    if x.shape != y.shape:
        raise ValueError(f"zx and zy must have same shape, got {x.shape} vs {y.shape}")
    return offline_plv(np.angle(x) - np.angle(y))


def offline_windowed_cepstrum_features(
    x: np.ndarray,
    win_len: int,
    hop_len: int,
    nfft: int,
    *,
    window: str | np.ndarray = "hann",
    eps: float = 1e-10,
    detrend: bool = False,
    min_quef_idx: int = 1,
    max_quef_idx: int | None = None,
    keep_per_window: bool = False,
) -> dict[str, Any]:
    x2 = _as_2d_time_major(x)

    acc = CepstrumFeatureAccumulator(
        nfft=nfft,
        window=window,
        eps=eps,
        detrend=detrend,
        min_quef_idx=min_quef_idx,
        max_quef_idx=max_quef_idx,
        keep_per_window=keep_per_window,
    )

    t = x2.shape[0]
    for start in range(0, t - win_len + 1, hop_len):
        end = start + win_len
        acc.update_window(x2[start:end, :])

    return acc.finalize()


@dataclass
class StreamingRunConfig:
    win_len: int = 128
    hop_len: int = 64
    nfft: int = 128
    window: str = "hann"
    eps: float = 1e-10
    detrend: bool = False
    keep_time_resolved: bool = False
    min_quef_idx: int = 1


class StreamingDeltaRunProcessor:
    """
    Stream processor for delta rows D[t, :].

    Signal definition aligned with current stability modules:
    - MSC/Cepstrum use |D[t, :]|.
    - PLV uses phase(D[t, :]) against 0-reference (same as existing baseline PLV formula).
    """

    def __init__(self, num_pages: int, config: StreamingRunConfig):
        self.num_pages = num_pages
        self.config = config

        self._mag_buffer = SlidingWindowBuffer(config.win_len, config.hop_len)
        self._msc = WelchMSCAccumulator(
            nfft=config.nfft,
            window=config.window,
            eps=config.eps,
            detrend=config.detrend,
            onesided=True,
        )
        self._plv = PLVAccumulator(num_channels=num_pages)
        self._cep = CepstrumFeatureAccumulator(
            nfft=config.nfft,
            window=config.window,
            eps=config.eps,
            detrend=config.detrend,
            min_quef_idx=config.min_quef_idx,
            keep_per_window=config.keep_time_resolved,
        )

        self._prev_mag_window: np.ndarray | None = None
        self._msc_pairs = 0

    def update_delta_frame(self, delta_frame: np.ndarray) -> None:
        row = np.asarray(delta_frame)
        if row.ndim != 1 or row.shape[0] != self.num_pages:
            raise ValueError(f"Expected delta frame shape ({self.num_pages},), got {row.shape}")

        mag = np.abs(row)
        phase = np.angle(row)

        self._plv.update_phase_diff(phase)

        ready_windows = self._mag_buffer.push(mag)
        for curr in ready_windows:
            self._cep.update_window(curr)
            if self._prev_mag_window is not None:
                self._msc.update_window(self._prev_mag_window, curr)
                self._msc_pairs += 1
            self._prev_mag_window = curr

    def finalize(self) -> dict[str, Any]:
        if self._msc_pairs == 0:
            wl, hl = self.config.win_len, self.config.hop_len
            raise ValueError(
                "Not enough windows for MSC. Need at least 2 complete magnitude windows "
                f"(win_len={wl}, hop_len={hl} => need T >= {wl + hl} frames along time; "
                "or reduce --win_len / --hop_len)."
            )

        msc = self._msc.finalize()
        msc_peak_snr_db_per_page = 10.0 * np.log10(
            (np.max(msc, axis=0) + self.config.eps)
            / (np.mean(msc, axis=0) + self.config.eps)
        )

        out = {
            "plv": {
                "plv_per_page": self._plv.finalize(),
            },
            "msc": {
                "msc_spectrum": msc,
                "msc_peak_snr_db_per_page": msc_peak_snr_db_per_page,
                "msc_peak_snr_db_median": float(np.median(msc_peak_snr_db_per_page)),
                "num_pairs": self._msc_pairs,
            },
            "cepstrum": self._cep.finalize(),
            "config": {
                "win_len": self.config.win_len,
                "hop_len": self.config.hop_len,
                "nfft": self.config.nfft,
                "window": self.config.window,
                "eps": self.config.eps,
                "detrend": self.config.detrend,
                "min_quef_idx": self.config.min_quef_idx,
            },
        }
        return out


def run_streaming_on_time_series(
    run_time_series: np.ndarray,
    *,
    win_len: int = 128,
    hop_len: int = 64,
    nfft: int | None = None,
    window: str = "hann",
    eps: float = 1e-10,
    detrend: bool = False,
    keep_time_resolved: bool = False,
    min_quef_idx: int = 1,
) -> dict[str, Any]:
    series = np.asarray(run_time_series)
    if series.ndim != 2:
        raise ValueError(f"Expected run_time_series with shape [T, N], got {series.shape}")

    cfg = StreamingRunConfig(
        win_len=win_len,
        hop_len=hop_len,
        nfft=win_len if nfft is None else nfft,
        window=window,
        eps=eps,
        detrend=detrend,
        keep_time_resolved=keep_time_resolved,
        min_quef_idx=min_quef_idx,
    )
    runner = StreamingDeltaRunProcessor(series.shape[1], cfg)
    for t in range(series.shape[0]):
        runner.update_delta_frame(series[t])

    return runner.finalize()


def save_streaming_results(results: dict[str, Any], output_prefix: str) -> None:
    """Save streaming outputs in .npz and lightweight .json summary."""

    msc = results["msc"]["msc_spectrum"]
    msc_peak = results["msc"]["msc_peak_snr_db_per_page"]
    plv = results["plv"]["plv_per_page"]

    np.savez_compressed(
        f"{output_prefix}.npz",
        msc_spectrum=msc,
        msc_peak_snr_db_per_page=msc_peak,
        plv_per_page=plv,
        cepstral_peak_idx_mean=results["cepstrum"]["cepstral_peak_idx_mean"],
        cepstral_peak_idx_var=results["cepstrum"]["cepstral_peak_idx_var"],
        cepstral_peak_idx_median=results["cepstrum"]["cepstral_peak_idx_median"],
    )

    summary = {
        "msc_peak_snr_db_median": results["msc"]["msc_peak_snr_db_median"],
        "plv_median": float(np.median(plv)),
        "cepstral_peak_idx_median": np.asarray(results["cepstrum"]["cepstral_peak_idx_median"]).tolist(),
        "config": results.get("config", {}),
    }
    with open(f"{output_prefix}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


import argparse
import numpy as np
import sys

def main():
    parser = argparse.ArgumentParser()
    # Input and output
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    # Streaming parameters (num_pages is inferred from input shape: array must be (time, num_pages))
    parser.add_argument("--win_len", type=int, default=128)
    parser.add_argument("--hop_len", type=int, default=64)
    parser.add_argument(
        "--nfft",
        type=int,
        default=0,
        help="FFT size; 0 means use win_len (default).",
    )
    parser.add_argument("--window", type=str, default="hann")
    parser.add_argument("--eps", type=float, default=1e-10)
    parser.add_argument("--detrend", type=bool, default=False)
    parser.add_argument("--keep_time_resolved", type=bool, default=False)
    parser.add_argument("--min_quef_idx", type=int, default=1)
    # Output
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--output_npz", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="streaming_results")

    args = parser.parse_args()

    run_time_series_transpose = np.load(args.input, mmap_mode='r')  
    run_time_series = run_time_series_transpose.T
    print(f"Loaded run time series with shape: {run_time_series.shape}")
    nfft_kw: int | None = None if args.nfft <= 0 else args.nfft
    results = run_streaming_on_time_series(
        run_time_series,
        win_len=args.win_len,
        hop_len=args.hop_len,
        nfft=nfft_kw,
        window=args.window,
        eps=args.eps,
        detrend=args.detrend,
        keep_time_resolved=args.keep_time_resolved,
        min_quef_idx=args.min_quef_idx,
    )
    print(f"Results: {results}")
    save_streaming_results(results, args.output)
    print(f"Saved results to {args.output}")

if __name__ == "__main__":
    main()