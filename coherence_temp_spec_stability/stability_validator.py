import numpy as np
from plv_calcolator import PLVStability
from magnitude_squared_coherence import MagnitudeSquaredCoherence
from cepstrum_stability import CepstrumStability
from streaming_metrics import run_streaming_on_time_series, save_streaming_results

# try:
#     from tqdm.auto import tqdm
# except ImportError:
#     print("tqdm not found, progress bars will be disabled. Install tqdm for better experience. Use - pip install tqdm.")
#     tqdm = None

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
        window_size: int = 32,
        window_step: int = 16,
        eps: float = 1e-10,
    ):
        """
        Args:
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
        drop_threshold: float = 0.2,
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
        
        msc_spectrum = self.msc_helper.compute_msc_streamlined(run_time_series)  # shape (F, N)
        print(f"MSC spectrum computed with shape: {msc_spectrum.shape}")

        peak_snr_db = self.msc_helper.compute_peak_snr(msc_spectrum)  # shape (N,)
        print(f"MSC peak SNR computed with shape: {peak_snr_db.shape}")

        peak_snr_db_median = float(np.median(peak_snr_db))
        print(f"MSC peak SNR median: {peak_snr_db_median}")

        print(f"\nDone computing MSC features.")
        print("------------------------------------------------")
        return {
            # 'msc_spectrum': msc_spectrum,
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
            # 'cepstrum': cepstrum,
            'cepstral_peak_idx_per_page': cepstral_peak_idx,
            'cepstral_peak_idx_median': cepstral_peak_idx_median,
        }


    @staticmethod
    def save_features_to_file(features: dict, filename: str = "run_features.npy"):
        np.save(filename, features, allow_pickle=True)


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
        print("\n================================================")
        print(f"Computing all stability features for run with shape: {run_time_series.shape}\n")
        print("Starting PLV feature computation...")
        plv_features = self.compute_plv_features(run_time_series)
        print("PLV features computed.\n")
        print("Starting MSC feature computation...")
        msc_features = self.compute_msc_features(run_time_series)
        print("MSC features computed.\n")
        print("Starting cepstrum feature computation...")
        cepstrum_features = self.compute_cepstrum_features(run_time_series)
        print("Cepstrum features computed.\n")
        print("All stability features computed.")
        print("================================================\n")
        combined_features = {
            'plv': plv_features,
            'msc': msc_features,
            'cepstrum': cepstrum_features,
        }

        return combined_features


def run_all_features_for_run(run_time_series: np.ndarray, window_size: int = 128, step_size: int = 64) -> dict:
    '''
        Computes PLV, MSC, and Cepstrum features for a WHOLE given run time-series using StabilityValidator.
        Args:
            run_time_series: Complex array [T_run, N] for the run to analyze.
        Returns:
            Dictionary of all computed features.
    '''
    sv = StabilityValidator(numPages = run_time_series.shape[1], window_size = window_size, window_step = step_size)
    sv.fit_plv_baseline(run_time_series) # Fit baseline on the same data for demonstration
    return sv.compute_all_features(run_time_series)


def run_all_features_512_windows(run_time_series: np.ndarray, window_size: int = 128, step_size: int = 64) -> list:  
    '''
        Computes PLV, MSC, and Cepstrum features on two separate episodes of the run time series of size 512 = 128 * 4, with
        overlap if needed (if size is less than 500) using StabilityValidator.
        Args:
            run_time_series: Complex array [T_run, N] for the run to analyze.
        Returns:
        List of both dictionaries of all computed features.
    '''
    num_pages = run_time_series.shape[1]
    sv = StabilityValidator(numPages = num_pages, window_size = window_size, window_step = step_size)

    sv.fit_plv_baseline(run_time_series[:512, :]) # Fit baseline on the first 512 samples

    features1 = sv.compute_all_features(run_time_series[:512, :])
    features2 = sv.compute_all_features(run_time_series[-512:, :])

    return [features1, features2]


def run_all_features_sliding_windows(run_time_series: np.ndarray, window_size: int = 128, step_size: int = 64) -> list:
    '''
        Computes PLV, MSC, and Cepstrum features on sliding windows of the run time series of size window_size with step step_size using StabilityValidator.
        MSC is calculated in sliding windows internally, so no need to do it separately here.
        Args:
            run_time_series: Complex array [T_run, N] for the run to analyze.
            window_size: Size of each window to analyze.
            step_size: Step size for sliding windows.
        Returns:
            List of dictionaries of computed features for each window.
    '''
    num_pages = run_time_series.shape[1]
    sv = StabilityValidator(numPages = num_pages, window_size = window_size, window_step = step_size)
    sv.fit_plv_baseline(run_time_series[:window_size, :]) # Fit baseline on the first window

    print("\n================================================")
    print(f"Computing all stability features for run with shape: {run_time_series.shape}\n")

    print("Starting MSC feature computation...")
    msc_features = sv.compute_msc_features(run_time_series)

    print("Starting PLV feature computation...")
    plv_features = {}
    for start in range(0, run_time_series.shape[0] - window_size + 1, step_size):
        end = min(start + window_size, run_time_series.shape[0])
        curr_plv_window = run_time_series[start:end, :]
        plv_features_window = sv.compute_plv_features(curr_plv_window)
        plv_features[f"wind_{start}_{end}"] = plv_features_window
    print("PLV features computed.\n")

    print("Starting cepstrum feature computation...")
    cepstrum_features = {}
    for start in range(0, run_time_series.shape[0] - window_size + 1, step_size):
        end = min(start + window_size, run_time_series.shape[0])
        curr_cepstrum_window = run_time_series[start:end, :]
        cepstrum_features_window = sv.compute_cepstrum_features(curr_cepstrum_window)
        cepstrum_features[f"wind_{start}_{end}"] = cepstrum_features_window
    print("Cepstrum features computed.\n")

    combined_features = {
        'plv': plv_features,
        'msc': msc_features,
        'cepstrum': cepstrum_features,
    }

    return combined_features


def run_all_features_streaming(
    run_time_series: np.ndarray,
    *,
    window_size: int = 128,
    step_size: int = 64,
    nfft: int | None = None,
    window: str = "hann",
    eps: float = 1e-10,
    detrend: bool = False,
    keep_time_resolved: bool = False,
    min_quef_idx: int = 1,
    output_prefix: str | None = None,
) -> dict:
    """
    Streaming/online feature extraction for delta rows D[t, :].

    This processes rows sequentially (memmap-friendly), updates online accumulators,
    and can optionally save .npz/.json outputs for a run.
    """
    results = run_streaming_on_time_series(
        run_time_series,
        win_len=window_size,
        hop_len=step_size,
        nfft=nfft,
        window=window,
        eps=eps,
        detrend=detrend,
        keep_time_resolved=keep_time_resolved,
        min_quef_idx=min_quef_idx,
    )
    if output_prefix:
        save_streaming_results(results, output_prefix)
    return results


# def main():
#     data_transpose = np.load(r"/Volumes/Extreme SSD/thesis/runs/19_01_26/A1/combined_data.npy", mmap_mode='r') # Load data from file memory-mapped
#     print(f"Loaded sample data with shape: {data_transpose.shape}")

#     data = data_transpose.T  # Transpose to [T, N]
#     print(f"Transposed data shape: {data.shape}")

#     T, N = data.shape

#     sv = StabilityValidator(numPages=N, window_size=32, window_step=16)

#     print("Fitting PLV baseline...")
#     sv.fit_plv_baseline(data) # Fit baseline on the same data for demonstration
#     print("PLV baseline fitted.")


#     plv_features = sv.compute_all_features(data)

#     print("Saving features to file...")
#     StabilityValidator.save_features_to_file(plv_features, filename="stability_features.npy")
#     print("================================================\n")
#     print("DONE.")


def main():
    for _ in range(1):
        X = "idle"
        data_transpose = np.load(f"/Volumes/Extreme SSD/thesis/runs/19_01_26/{X}/combined_data.npy", mmap_mode='r') # Load data from file memory-mapped
        print(f"Loaded sample data with shape: {data_transpose.shape}")

        data = data_transpose.T  # Transpose to [T, N]
        print(f"Transposed data shape: {data.shape}")

        print("Running all features for the whole run...")
        full_run = run_all_features_for_run(data, window_size=128, step_size=64)

        print("Running all features for 512-sample windows...")
        windows_512 = run_all_features_512_windows(data, window_size=128, step_size=64)
        
        print("Running all features for sliding windows...")
        sliding_windows_128_64 = run_all_features_sliding_windows(data, window_size=128, step_size=64)    

        features = {
            'full_run': full_run,
            '512_windows': windows_512,
            'sliding_windows_128_64': sliding_windows_128_64,
        }

        print("Saving features to file...")
        StabilityValidator.save_features_to_file(features, filename=f"stability_features_{X}.npy")
        print("================================================\n")
        print(f"DONE {X} - features saved to stability_features_{X}.npy\n")


def cli_run_validator() -> None:
    """CLI for running stability (streaming) metrics on a matrix .npy (e.g. raw matrix from raw_matrix_builder)."""
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Run PLV/MSC/Cepstrum streaming metrics on a time-series matrix."
    )
    parser.add_argument(
        "matrix_npy",
        nargs="?",
        default=None,
        help="Path to .npy matrix; shape (num_pages, num_frames) or (num_frames, num_pages) — auto-detected",
    )
    parser.add_argument("--window-size", type=int, default=128, help="Window size for streaming")
    parser.add_argument("--step-size", type=int, default=64, help="Step size for streaming")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory for output .npz and .json")
    parser.add_argument("--prefix", type=str, default="raw", help="Output file prefix (output_dir/prefix.npz, .json)")
    args = parser.parse_args()

    if not args.matrix_npy:
        main()
        return

    data = np.load(args.matrix_npy, mmap_mode="r")
    # Expect (num_pages, num_frames); streaming wants (T, N) = (num_frames, num_pages)
    if data.shape[0] < data.shape[1]:
        data = data.T
    data = np.asarray(data)
    output_prefix = os.path.join(args.output_dir, args.prefix)
    os.makedirs(args.output_dir, exist_ok=True)
    run_all_features_streaming(
        data,
        window_size=args.window_size,
        step_size=args.step_size,
        output_prefix=output_prefix,
    )
    print(f"Done. Outputs: {output_prefix}.npz, {output_prefix}.json")


if __name__ == "__main__":
    cli_run_validator()
