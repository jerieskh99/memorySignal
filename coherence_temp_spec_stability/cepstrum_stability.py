# coherence_temp_spec_stability/cepstrum_stability.py
import numpy as np


class CepstrumStability:
    """
    Cepstral stability calculator.

    Computes:
      - Real cepstrum per page.
      - Dominant cepstral peak index (quefrency) per page.
    """

    def __init__(self, eps: float = 1e-10):
        """
        Args:
            eps: Small constant to stabilize log-magnitude.
        """
        self.epsilon = eps

    def compute_cepstrum(self, time_series: np.ndarray) -> np.ndarray:
        """
        Compute real cepstrum for each page over the full time-series.

        Args:
            time_series: Complex or real array [T, N].

        Returns:
            cepstrum: Real array [Q, N],
                      Q = number of quefrency bins (typically T).
        """
        # Added because cepstrum accept real inputs
        if np.iscomplexobj(time_series):
            time_series = np.abs(time_series)

        fft = np.fft.rfft(time_series, axis=0)
        log_mag = np.log(np.abs(fft) + self.epsilon)
        cepstrum = np.fft.irfft(log_mag, axis=0)
        return cepstrum



    def compute_cepstral_peak(
        self,
        cepstrum: np.ndarray,
        *,
        min_quef_idx: int = 1,
    ) -> np.ndarray:
        """
        Find dominant cepstral peak index for each page.

        Args:
            cepstrum: Real array [Q, N].
            min_quef_idx: Minimum quefrency index to consider (to skip DC).

        Returns:
            peak_idx: Integer array [N] with the index of the strongest peak
                      |cepstrum[q, n]| for each page n.
        """
        Q, N = cepstrum.shape
        peaks = np.zeros(N, dtype=int)
        for n in range(N):
            c = np.abs(cepstrum[:, n])
            c_search = c[min_quef_idx:]
            local_max_idx = np.argmax(c_search)
            peaks[n] = local_max_idx + min_quef_idx
        
        return peaks
