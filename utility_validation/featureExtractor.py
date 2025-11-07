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
    
    def extract_phase_features(self, phase_t):
        T = len(phase_t)

        c = np.cos(phase_t).mean(axis=0)
        s = np.sin(phase_t).mean(axis=0)
        R = np.sqrt(c**2 + s**2)
        phase_var = 1 - R
        consistency_adjacent = np.mean(np.abs(np.diff(phase_t, axis=0)), axis=0) if T>1 else 0.0

        features = np.concatenate((c, s, phase_var, consistency_adjacent))
        return features
    
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