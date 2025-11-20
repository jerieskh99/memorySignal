# coherence_temp_spec_stability/stability_validator.py
import numpy as np
from utils.randPageSelector import Selector
from coherence_temp_spec_stability.plv_calcolator import PLVStability
from coherence_temp_spec_stability.magnitude_squared_coherence import MagnitudeSquaredCoherence
from coherence_temp_spec_stability.cepstrum_stability import CepstrumStability


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
        pathBitmap: str,
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
        self.pathBitmap = pathBitmap
        self.numPages = numPages

        self.window_size = window_size
        self.window_step = window_step
        self.eps = eps

        self.selector = Selector(pathBitmap, numPages)

        self.plv_helper = PLVStability()
        self.msc_helper = MagnitudeSquaredCoherence(
            window_size=window_size,
            window_step=window_step,
            eps=eps,
        )
        self.cepstrum_helper = CepstrumStability(eps=eps)

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
        
        msc_spectrum = self.msc_helper.compute_msc(run_time_series)  # shape (F, N)
        peak_snr_db = self.msc_helper.compute_peak_snr(msc_spectrum)  # shape (N,)
        peak_snr_db_median = float(np.median(peak_snr_db))

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
        
        cepstrum = self.cepstrum_helper.compute_cepstrum(run_time_series)
        cepstral_peak_idx = self.cepstrum_helper.compute_cepstral_peak(cepstrum)
        cepstral_peak_idx_median = float(np.median(cepstral_peak_idx))

        return {
            'cepstrum': cepstrum,
            'cepstral_peak_idx_per_page': cepstral_peak_idx,
            'cepstral_peak_idx_median': cepstral_peak_idx_median,
        }

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
