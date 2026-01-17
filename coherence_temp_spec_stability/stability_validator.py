import numpy as np
from plv_calcolator import PLVStability
from magnitude_squared_coherence import MagnitudeSquaredCoherence
from cepstrum_stability import CepstrumStability


class StabilityValidator:
    """
    High-level orchestrator for Pillar 3 — Temporal & Spectral Coherence.

    Responsibilities:
      - Hold a PLV baseline (from clean data).
      - For any given run (clean / infected / other host), compute:
          * PLV-based stability features.
          * Magnitude-Squared Coherence (MSC) features.
          * Cepstrum-based stability features.
      - Aggregate them into a single feature dictionary per run.
    """

    def __init__(
        self,
        numPages: int,
        *,
        window_size: int = 128,
        window_step: int = 64,
        eps: float = 1e-10,
    ):
        """
        Args:
            pathBitmap: Path to bitmap used by Selector to filter active pages.
            numPages: Total number of pages (before bitmap filtering).
            window_size: Window length for MSC calculations.
            window_step: Window hop size for MSC calculations.
            eps: Small constant shared by coherence-related modules.
        """
        self.numPages = numPages

        self.window_size = window_size
        self.window_step = window_step
        self.eps = eps
        print("Modules initialized:")
        
        self.plv_helper = PLVStability()
        print(f" - {self.plv_helper.__class__.__name__}")

        self.msc_helper = MagnitudeSquaredCoherence(
            window_size=window_size,
            window_step=window_step,
            eps=eps,
        )
        print(f" - {self.msc_helper.__class__.__name__}")

        self.cepstrum_helper = CepstrumStability(eps=eps)
        print(f" - {self.cepstrum_helper.__class__.__name__}")
        print("================================================")

    # -------- PLV (baseline + per-run) --------

    def fit_plv_baseline(self, clean_time_series: np.ndarray) -> np.ndarray:
        """
        Fit and store PLV baseline from clean time-series data.

        Args:
            clean_time_series: Complex array [T_clean, N].

        Returns:
            baseline_plv: Real array [N].
        """
        if clean_time_series.ndim != 2:
            raise ValueError("Expected 2D [T_clean, N] clean_time_series")
        
        self.plv_helper.fit_baseline(clean_time_series)
        return self.plv_helper.get_baseline_plv()

    def compute_plv_features(
        self,
        run_time_series: np.ndarray,
        *,
        drop_threshold: float = -0.2,
        normal_threshold: float = 0.7,
        perfect_threshold: float = 1.0,
    ) -> dict:
        """
        Compute PLV-based features for a run using the stored baseline.

        Args:
            run_time_series: Complex array [T_run, N].
            drop_threshold: Median PLV drop threshold for global anomaly flag.
            normal_threshold: PLV threshold for "normal" stability.
            perfect_threshold: PLV threshold for "perfect" stability.

        Returns:
            Dictionary as returned by PLVStability.evaluate_run().
        """
        if self.plv_helper.get_baseline_plv() is None:
            raise RuntimeError("PLV baseline not set. Call fit_plv_baseline() first.")
        
        if run_time_series.ndim != 2:
            raise ValueError("Expected 2D [T_run, N] run_time_series")
        
        baseline = self.plv_helper.get_baseline_plv()
        if baseline.shape[0] != run_time_series.shape[1]:
            raise ValueError(
                f"Baseline PLV has N={baseline.shape[0]} pages but run has N={run_time_series.shape[1]}"
            )

        plv_features = self.plv_helper.evaluate_run(
            run_time_series,
            drop_threshold=drop_threshold,
            normal_threshold=normal_threshold,
            perfect_threshold=perfect_threshold,
        )

        return plv_features 

    # -------- Magnitude-Squared Coherence --------

    def compute_msc_features(self, run_time_series: np.ndarray) -> dict:
        """
        Compute MSC spectrum and its derived features for a run.

        Args:
            run_time_series: Complex array [T_run, N].

        Returns:
            Dictionary with at least:
              - 'msc_spectrum': [F, N]
              - 'msc_peak_snr_db_per_page': [N]
              - 'msc_peak_snr_db_median': float
        """
        if run_time_series.ndim != 2:
            raise ValueError("Expected 2D [T_run, N] run_time_series")
        
        if run_time_series.shape[1] != self.numPages:
            raise ValueError(
                f"Expected run_time_series with N={self.numPages} pages, got N={run_time_series.shape[1]}"
            )
        
        print("\n------------------------------------------------")
        print(f"Computing MSC features for run with shape: {run_time_series.shape}\n")
        
        msc_spectrum = self.msc_helper.compute_msc(run_time_series)  # shape (F, N)
        print(f"MSC spectrum computed with shape: {msc_spectrum.shape}")

        peak_snr_db = self.msc_helper.compute_peak_snr(msc_spectrum)  # shape (N,)
        print(f"MSC peak SNR computed with shape: {peak_snr_db.shape}")

        peak_snr_db_median = float(np.median(peak_snr_db))
        print(f"MSC peak SNR median: {peak_snr_db_median}")

        print(f"\nDone computing MSC features.")
        print("------------------------------------------------")
        return {
            'msc_spectrum': msc_spectrum,
            'msc_peak_snr_db_per_page': peak_snr_db,
            'msc_peak_snr_db_median': peak_snr_db_median,
        }

    # -------- Cepstrum & Cepstral Peak --------

    def compute_cepstrum_features(self, run_time_series: np.ndarray) -> dict:
        """
        Compute cepstrum and cepstral-peak features for a run.

        Args:
            run_time_series: Complex array [T_run, N].

        Returns:
            Dictionary with at least:
              - 'cepstrum': [Q, N]
              - 'cepstral_peak_idx_per_page': [N]
              - 'cepstral_peak_idx_median': float
        """
        if run_time_series.ndim != 2:
            raise ValueError("Expected 2D [T_run, N] run_time_series")
        
        print("\n------------------------------------------------")
        print(f"Computing cepstrum features for run with shape: {run_time_series.shape}\n")

        cepstrum = self.cepstrum_helper.compute_cepstrum(run_time_series)
        print(f"Cepstrum computed with shape: {cepstrum.shape}")

        cepstral_peak_idx = self.cepstrum_helper.compute_cepstral_peak(cepstrum)
        print(f"Cepstral peak indices computed with shape: {cepstral_peak_idx.shape}")

        cepstral_peak_idx_median = float(np.median(cepstral_peak_idx))
        print(f"Cepstral peak index median: {cepstral_peak_idx_median}")

        print("\nDone computing cepstrum features.")
        print("------------------------------------------------\n")

        return {
            'cepstrum': cepstrum,
            'cepstral_peak_idx_per_page': cepstral_peak_idx,
            'cepstral_peak_idx_median': cepstral_peak_idx_median,
        }


    def save_features_to_file(features: dict, filename: str = "run_features.npy"):
        """
        Save computed features dictionary to a .npy file.

        Args:
            features: Dictionary of features to save.
            filename: Output filename.
        """
        np.save(filename, features)
        print(f"Features saved to {filename}")


    # -------- Combined interface --------

    def compute_all_features(self, run_time_series: np.ndarray) -> dict:
        """
        Compute all stability-related features for a run in one call.

        Args:
            run_time_series: Complex array [T_run, N].

        Returns:
            features: Dict containing:
              - 'plv'              : dict (PLV features)
              - 'msc'              : dict (MSC features)
              - 'cepstrum'         : dict (cepstrum features)
              - Optional derived summary statistics per run.
        """
        plv_features = self.compute_plv_features(run_time_series)
        msc_features = self.compute_msc_features(run_time_series)
        cepstrum_features = self.compute_cepstrum_features(run_time_series)

        combined_features = {
            'plv': plv_features,
            'msc': msc_features,
            'cepstrum': cepstrum_features,
        }

        return combined_features



def main():
    data_transpose = np.load("/Users/jeries/Desktop/projects/thesis/memorySignal/mem_sig/data/combined_data.npy", mmap_mode='r') # Load data from file memory-mapped
    print(f"Loaded sample data with shape: {data_transpose.shape}")

    data = data_transpose.T  # Transpose to [T, N]
    print(f"Transposed data shape: {data.shape}")

    num_pages_data = data.shape[1]

    sv = StabilityValidator(numPages=num_pages_data, window_size=32, window_step=16)

    sv.fit_plv_baseline(data) # Fit baseline on the same data for demonstration

    plv_features = sv.compute_all_features(data)

    sv.save_features_to_file(plv_features, filename="run_features.npy")


if __name__ == "__main__":
    main()