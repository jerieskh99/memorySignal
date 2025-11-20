import numpy as np


class MagnitudeSquaredCoherence:
    """
    Magnitude-Squared Coherence (MSC) calculator.

    Computes:
      - MSC spectra between adjacent overlapping windows per page.
      - Coherence-peak SNR (in dB) per page.
    """

    def __init__(
        self,
        window_size: int = 128,
        window_step: int = 64,
        eps: float = 1e-10,
    ):
        """
        Args:
            window_size: Length L of each time window.
            window_step: Step between consecutive window starts.
            eps: Small constant to avoid division by zero.
        """
        self.window_size = window_size
        self.window_step = window_step
        self.epsilon = eps

    def slice_windows(self, time_series: np.ndarray) -> np.ndarray:
        """
        Slice a time-series into overlapping windows along the time axis.

        Args:
            time_series: Array [T, N], typically real non-negative magnitudes.

        Returns:
            windows: Array [W, L, N]
              W = number of windows
              L = window_size
        """
        if time_series.ndim != 2:
            raise ValueError("Expected 2D [T, N] time_series")
        
        T, N = time_series.shape
        start_indices = np.arange(0, T - self.window_size + 1, self.window_step)
        W = len(start_indices)
        windows = np.zeros((W, self.window_size, N), dtype=time_series.dtype)

        for i, start in enumerate(start_indices):
            windows[i] = time_series[start: start + self.window_size, :]
        
        return windows

    def compute_pair_msc(
        self,
        win_x: np.ndarray,
        win_y: np.ndarray,
    ) -> np.ndarray:
        """
        Compute magnitude-squared coherence between two windows.

        Args:
            win_x: Windowed data [L, N].
            win_y: Windowed data [L, N].

        Returns:
            msc: MSC spectrum [F, N],
                 F = number of rFFT bins = L//2 + 1.
        """
        if win_x.shape != win_y.shape:
            raise ValueError("win_x and win_y must have the same shape")
        
        if win_x.ndim != 2:
            raise ValueError("Expected 2D [L, N] windows")
        
        if win_y.ndim != 2:
            raise ValueError("Expected 2D [L, N] windows")
        
        fft_x = np.fft.rfft(win_x, axis=0)
        fft_y = np.fft.rfft(win_y, axis=0)
        power_spectra_x = np.abs(fft_x) ** 2
        power_spectra_y = np.abs(fft_y) ** 2

        cross_spectra = fft_x * np.conj(fft_y)
        numerator = np.abs(cross_spectra) ** 2
        denominator = power_spectra_x * power_spectra_y + self.epsilon

        coherence = numerator / denominator
        return coherence

    def compute_msc(self, time_series: np.ndarray) -> np.ndarray:
        """
        Compute MSC between all adjacent windows of a time-series.

        Args:
            time_series: Real or complex array [T, N].
                         (Magnitude is typically taken before computing MSC.)

        Returns:
            msc_mean: Real array [F, N] with mean MSC over adjacent window pairs.
        """
        if time_series.ndim != 2:
            raise ValueError("Expected 2D [T, N] time_series")
        
        if np.iscomplexobj(time_series):
            mags = np.abs(time_series) # shape (T, N)
        else:
            mags = time_series # shape (T, N)
        
        windows = self.slice_windows(mags)
        W,L , N = windows.shape

        if windows.shape != (W, self.window_size, N):
            raise ValueError("Window slicing produced unexpected shape")
        
        if W < 2:
            raise ValueError("Not enough windows to compute MSC; increase time-series length or decrease window size/step.")
        
        pair_wise_msc = np.empty((W - 1, L//2 + 1, N))
        for w in range(W - 1):
            pair_wise_msc[w] = self.compute_pair_msc(windows[w], windows[w+1])
        
        msc_mean = np.mean(np.stack(pair_wise_msc, axis=0), axis=0)

        return msc_mean


    def compute_peak_snr(self, msc: np.ndarray) -> np.ndarray:
        """
        Compute coherence-peak SNR (in dB) per page.

        Args:
            msc: MSC spectrum [F, N].

        Returns:
            peak_snr_db: Real array [N] with SNR in dB for each page.
        """
        peaks = np.max(msc, axis=0) # shape (N,)
        noise = np.mean(msc, axis=0)

        snr_db = 10.0 * np.log10((peaks + self.epsilon)/( noise + self.epsilon)) # shape (N,)

        return snr_db
