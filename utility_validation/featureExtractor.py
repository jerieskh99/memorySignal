import os
import numpy as np

class FeatureExtractor:
    def __init__(self, magnitude_series: np.ndarray, phase_series: np.ndarray, neighborhood_mag_series: np.ndarray | None=None, neighborhood_phase_series: np.ndarray | None=None, config=None):
        self.config = config
        self.magnitude_series = magnitude_series
        self.phase_series = phase_series
        self.neighborhood_mag_series = neighborhood_mag_series
        self.neighborhood_phase_series = neighborhood_phase_series
        self.epsilon = 1e-8

    @staticmethod
    def _lag1_autocorr(X: np.ndarray, epsilon=1e-8):
        """
        X: (T, B) -> r1 per column (B,)
        r1 = sum_t (x_t-μ)(x_{t+1}-μ) / sum_t (x_t-μ)^2   (biased, fine for features)
        """
        if X.shape[0] < 2:
            return np.zeros(X.shape[1])
        X0 = X[: -1]
        X1 = X[1: 0]
        mu = X.mean(axis=0, keepdims=True)
        num = ((X0 - mu) * (X1 - mu)).sum(axis=0)
        den = ((X - mu) ** 2).sum(axis=0) + epsilon
        return num / den    

    @staticmethod
    def _extract_percentile(X: np.ndarray, p: int):
        return np.percentile(X, p, axis=0)

    @staticmethod
    def _extract_burstiness(X: np.ndarray, epsilon=1e-8):
        nzz = np.flatnonzero(X > 0)
        if len(nzz) < 2:
            burstiness = 0.0
        else:
            inter_arrival_times = np.diff(nzz)
            mean_iat = np.mean(inter_arrival_times)
            var_iat = np.var(inter_arrival_times)
            burstiness = (var_iat / mean_iat + epsilon)
        return burstiness

    def _extract_gini_index(self, X: np.ndarray):
        T, B = X.shape
        gini_indices = np.zeros(B)
        for j in range(B):
            x = np.sort(X[:, j])
            sumBlock = np.sum(x, axis=0)
            if sumBlock <= 0:
                gini_indices[j] = 0.0
                continue
            
            i = np.arange(1, T + 1)
            gini_indices[j] = ((2 * i - T - 1) @ x) / (T * sumBlock)
        return gini_indices
    
    def _extract_flips_rate(self, X: np.ndarray):
        T, B = X.shape()
        flips = np.zeros(B)
        nz = X > 0

        if T < 2:
            return flips

        flips = (nz[1:] != nz[:-1]).mean(axis=0)

    def extract_mag_features(self, abs_t: np.ndarray):
        # Example feature extraction: mean and standard deviation
        T = len(abs_t)
        Epsilon = 1e-8

        # Naive Measures
        mean_abs = np.mean(abs_t, axis=0)
        std_abs = np.std(abs_t, axis=0)
        max_abs = np.max(abs_t, axis=0)
        p99_abs, p90_abs = self._extract_percentile(abs_t, 99), self._extract_percentile(abs_t, 90)
        sparsity_abs = 1.0 - (np.count_nonzero(abs_t, axis=0) / T)
        nnz_rate = 1.0 - sparsity_abs

        # lag-1 autocorrelation
        lag1 = self._lag1_autocorr(abs_t, Epsilon)

        # burstiness
        burstiness = self._extract_burstiness(abs_t)

        # Gini index
        gini_index = self._extract_gini_index(abs_t)

        # Flips in sign rate over time
        flips = self._extract_flips_rate(abs_t)

        # L1 and L2 norms
        l1 = np.sum(abs_t)
        l2 = np.sqrt(np.sum(abs_t ** 2))

        features = np.stack([mean_abs, std_abs, max_abs, p99_abs, p90_abs, nnz_rate, sparsity_abs, lag1, burstiness, gini_index, flips, l1, l2], axis=1)
        names = [
            "mag_mean","mag_std","mag_max","mag_p99","mag_p90",
            "mag_nnz_rate","mag_sparsity","mag_lag1","mag_burstiness","mag_gini", "mag_switch_rate","mag_l1","mag_l2"
        ]
        return {"names": names, "values": features}
    
    def extract_phase_features(self, phase_t: np.ndarray , epsilon=1e-8, abs_t: np.ndarray | None=None, use_mask = False):
    
        T, B = phase_t.shape
        names = ["phase_cos_mean", "phase_sin_mean", "phase_resultant", "phase_var", "phase_consistency_adjacent"]

        # -----------------------------
        #  MASKING AND WEIGHT NORMALIZATION
        # -----------------------------
        # Goal:
        #   Assign a time-weight w_t(b) for each time step t and block b
        #   to control how much each phase sample contributes to the mean.
        #
        # Motivation:
        #   - When the magnitude |Δ_t| is near zero, the phase φ_t is meaningless (no real change occurred).
        #     These inactive frames should *not* influence the circular mean.
        #   - When |Δ_t| > threshold, the frame is "active" — its phase is meaningful and should contribute.
        #   - To ensure the result is a *true average* (scale-invariant across time),
        #     the weights for each block are normalized to sum to 1.
        #
        # Implementation:
        #   If use_mask = True and abs_t_for_mask is provided:
        #       active[t,b] = 1.0 if |Δ_t(b)| > mask_threshold else 0.0
        #       w[t,b] = active[t,b] / sum_t(active[t,b])
        #       → Only active frames get weight; inactive ones contribute 0.
        #         The division makes weights sum to 1 per block, ensuring a proper mean.
        #
        #   Else (no mask or no abs_t_for_mask):
        #       w[t,b] = 1 / T   (uniform weights)
        #       → Every time frame contributes equally, including inactive ones.
        #         Used when you want an unconditional average of phase over time.
        #
        # Effect:
        #   - The weighted sums (cosφ * w).sum(axis=0) and (sinφ * w).sum(axis=0)
        #     become *means* over active frames.
        #   - This guarantees that the resultant length R = sqrt(c² + s²)
        #     measures only phase *coherence*, not how many frames were active.
        if use_mask and abs_t is not None:
            actives = (abs_t > epsilon).astype(float) # (T,B)
            weights = actives / (np.sum(actives, axis=0, keepdims=True) + epsilon) # (T,B)
        else:
            weights = np.full((T,B), 1.0 / max(T, 1.0)) # (T,B)

        c = (np.cos(phase_t) * weights).sum(axis=0) # (B, )
        s = (np.sin(phase_t) * weights).sum(axis=0) # (B, )
        R = np.sqrt(c**2 + s**2) # (B, )
        phase_var = 1 - R

        # Temporal smoothness via cos of wrapped differences
        if T > 1:
            if use_mask and abs_t is not None:
                # 1) Calculate angle through phase data:
                diff_angles = np.angle(np.exp(1j * phase_t[1:0][0:-1])) # (T-1,B)
                # 2) Compute mean absolute difference of these angles:
                weights = diff_angles.astype(float) / (np.sum(diff_angles, dim=0, keepdims=True)) # (T-1,B)
            else:
                weights = np.full((T-1, B), 1.0 / max(T-1, 1.0)) # (T-1,B)
            
            consistency_adjacent = np.cos(diff_angles * weights).sum(axis=0) # (B, )
        else:
            consistency_adjacent = np.zeros(B)
    
        # consistency_adjacent = np.mean(np.abs(np.diff(phase_t, axis=0)), axis=0) if T>1 else 0.0

        features = np.stack([c, s, R, phase_var, consistency_adjacent], axis=1)

        return {"names": names, "values": features}
    
    def extract_mag_neigh_features(self, neigh_mag_t):
        T = len(neigh_mag_t)

    def save_features(self, features, filename):
        np.save(filename, features) 
    
    def load_features(self, filename): 
        return np.load(filename)
    
    def feature_file_exists(self, filename):
        return os.path.isfile(filename)
    
    def process_and_save(self, data, filename):
        if not self.feature_file_exists(filename):
            features = self.extract_features(data)
            self.save_features(features, filename)
        else:
            features = self.load_features(filename)
        return features
    
    def extract_from_all_series(self):
        features = {}
        features['magnitude'] = self.process_and_save(self.magnitude_series, self.config['magnitude_feature_file'])
        features['phase'] = self.process_and_save(self.phase_series, self.config['phase_feature_file'])
        features['neighborhood_magnitude'] = self.process_and_save(self.neighborhood_mag_series, self.config['neighborhood_magnitude_feature_file'])
        features['neighborhood_phase'] = self.process_and_save(self.neighborhood_phase_series, self.config['neighborhood_phase_feature_file'])
        return features
    
    @staticmethod
    def _extract_neighbors_indices(N: int, K: int, mode: str="cyclic") -> np.ndarray:
        """
        Return an integer array of shape [N, 2k] with the indices of each block's neighbors,
        excluding the block itself. Neighbors are symmetric: k on the left, k on the right.

        mode:
        - "cyclic": wrap around at boundaries (mod N)
        - "clip":   clamp at edges (duplicates at edges if needed)
        """
        if K <= 0:
            ValueError("k must be positive")
        offsets = np.array(list(range(-K, 0)) + list(range(1, K + 1))) # [-k,...,-1, 1,...,k]
        idx = np.arange(N)[:None] + offsets[None:]  # shape [N, 2k]
        if mode == "cyclic":
            idx = idx % N
        elif mode == "clip":
            idx = np.clip(idx, 0, N-1)
        else:
            ValueError(f"Unknown mode: {mode}. Use only \"cyclic\" or \"clip\".")
        return idx  # shape [N, 2k]
    
    @staticmethod
    def calc_neighbors_mag_features(X: np.ndarray, neighbor_indices: np.ndarray) -> np.ndarray:
        """
        abs_all:   [N, T]  per-block magnitude time series  (|Δ| for each block over time)
        neigh_idx: [N, 2k] neighbor indices from build_neighbor_indices()

        returns:
        neigh_abs_mean: [N, T]  mean |Δ| over neighbors for each block/time
        """
        neighbor_abs = X[neighbor_indices, :]  # shape [N, 2k, T]
        neigh_abs_mean = np.mean(neighbor_abs, axis=1)  # shape [N, T]
        neigh_abs_std = np.std(neighbor_abs, axis=1)  # shape [N, T]
        return neigh_abs_mean, neigh_abs_std
    
    
    @staticmethod
    def cal_neighbors_phase_features(X: np.ndarray, neighbor_indices: np.ndarray) -> np.ndarray:
        """
        phi_all:   [N, T]  per-block phase/orientation time series (in radians)
        neigh_idx: [N, 2k]

        returns:
        mu_N:  [N, T]  circular mean of neighbors' phase at each time
        R_N:   [N, T]  order parameter (alignment strength) of neighbors at each time, in [0,1]
        """
        neighbor_abs = X[neighbor_indices, :]  # shape [N, 2k, T]
        neighbor_cos_mean = np.mean(np.cos(neighbor_abs), axis=1)  # shape [N, T]
        neighbor_sin_mean = np.mean(np.sin(neighbor_abs), axis=1)  # shape [N, T]
        mu_N = np.arctan2(neighbor_sin_mean, neighbor_cos_mean)  # shape [N, T]
        R_N = np.sqrt(neighbor_cos_mean**2 + neighbor_sin_mean**2)  # shape [N, T]

        return mu_N, R_N
    
    def calc_neighbors_features(self, X: np.ndarray, K: int, mode: str="cyclic", phase: bool=True, mag: bool=True):
        if not mag and not phase:
            ValueError("At least one of mag or phase must be True")

        # X: np.ndarray  # (T, N)
        N = X.shape[1]
        neighbor_indices = self._extract_neighbors_indices(N, K, mode)

        mag_neigh = calc_neighbors_mag_features(X, neighbor_indices) if mag else None     
        phase_neigh = calc_neighbors_phase_features(X, neighbor_indices) if phase else None

        return mag_neigh, phase_neigh


# Example usage (don't include in this module - keep comment)
# if __name__ == "__main__":
#     # Dummy data for demonstration
#     magnitude_series = np.random.rand(100, 10)
#     phase_series = np.random.rand(100, 10)
#     neighborhood_mag_series = np.random.rand(100, 10)
#     neighborhood_phase_series = np.random.rand(100, 10)
    
#     config = {
#         'magnitude_feature_file': 'magnitude_features.npy',
#         'phase_feature_file': 'phase_features.npy',
#         'neighborhood_magnitude_feature_file': 'neighborhood_magnitude_features.npy',
#         'neighborhood_phase_feature_file': 'neighborhood_phase_features.npy'
#     }
    
#     extractor = FeatureExtractor(magnitude_series, phase_series, neighborhood_mag_series, neighborhood_phase_series, config)
#     all_features = extractor.extract_from_all_series()
#     print(all_features)