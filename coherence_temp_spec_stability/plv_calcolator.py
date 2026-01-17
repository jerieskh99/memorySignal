import numpy as np

class PLVStability:
    def __init__(self, baseline_plv: np.ndarray = None, epsilon = 0.01):
        self.baseline_plv = baseline_plv
        self.epsilon = epsilon
    
    @staticmethod
    def _getPhaseOfComplexSignal(data: np.ndarray) -> np.ndarray:
        return np.angle(data)
    
    @staticmethod
    def _computePLV(data: np.ndarray) -> np.ndarray:
        exponents = np.exp(1j * PLVStability._getPhaseOfComplexSignal(data))
        plv = np.abs(exponents.mean(axis=0))
        return plv
    
    # --------- 1) CLEAN BASELINE PHASE ---------
    def fit_baseline(self, clean_time_series: np.ndarray):
        """
        Compute and store baseline PLV from clean data.
        """
        if clean_time_series.ndim != 2:
            raise ValueError("Expected 2D [T, N] clean_time_series")
        self.baseline_plv = self._computePLV(clean_time_series)
        return self.baseline_plv
    
    def get_baseline_plv(self):
        return self.baseline_plv

    # --------- 2) EVALUATION PHASE ---------
    def evaluate_run(self, curr_time_series: np.ndarray, drop_threshold = -0.2, normal_threshold = 0.7, perfect_threshold = 1.00):
        if self.baseline_plv is None:
            raise RuntimeError("Call fit_baseline() first with clean data to set baseline PLV.")
        
        if self.baseline_plv.shape != (curr_time_series.shape[1],):
            raise ValueError("Baseline PLV shape does not match current data shape.")
        
        if curr_time_series.ndim != 2:
            raise ValueError("Expected 2D [T, N] curr_time_series")
        
        # --- PLV for current run ---
        current_plv = self._computePLV(curr_time_series)      # shape (N,)

        # per-page drop: current - baseline
        delta_plv = current_plv - self.baseline_plv           # shape (N,)

        # medians (for reporting)
        median_clean = float(np.median(self.baseline_plv))
        median_curr = float(np.median(current_plv))
        median_drop = float(np.median(delta_plv))             # typically <= 0 if things got worse

        # global anomaly flag based on median drop
        is_anomaly = median_drop <= drop_threshold

        # --- per-page statuses (informative labels) ---
        statuses = []
        for p, d in zip(current_plv, delta_plv):
            if d <= drop_threshold: # recently flipped...
                status = "anomalous_drop"     # lost too much stability vs baseline
            elif p < 0.4:
                status = "very_weak_stability"
            elif p < normal_threshold:
                status = "moderate_stability"
            elif p < perfect_threshold:
                status = "high_stability"
            else:
                status = "perfect_stability"
            statuses.append(status)

        return {
            "baseline_plv": self.baseline_plv,
            "current_plv": current_plv,
            "delta_plv": delta_plv,
            "median_baseline": median_clean,
            "median_current": median_curr,
            "median_drop": median_drop,
            "is_anomaly": is_anomaly,
            "statuses": statuses,            # per-page labels
        }