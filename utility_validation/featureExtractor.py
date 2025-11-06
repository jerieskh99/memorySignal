import os
import numpy as np

class FeatureExtractor:
    def __init__(self, magnitude_series, phase_series, neighborhood_mag_series, neighborhood_phase_series, config):
        self.config = config
        self.magnitude_series = magnitude_series
        self.phase_series = phase_series
        self.neighborhood_mag_series = neighborhood_mag_series
        self.neighborhood_phase_series = neighborhood_phase_series


    def extract_mag_features(self, abs_t):
        # Example feature extraction: mean and standard deviation
        T = len(abs_t)
        Epsilon = 1e-8


        mean_abs = np.mean(abs_t, axis=0)
        std_abs = np.std(abs_t, axis=0)
        max_abs = np.max(abs_t, axis=0)
        p99_abs = np.percentile(abs_t, 99, axis=0)
        p90_abs = np.percentile(abs_t, 90, axis=0)
        sparsity_abs = 1.0 - (np.count_nonzero(abs_t, axis=0) / T)
        nnz_rate = 1.0 - sparsity_abs
        lag1 = np.corrcoef(abs_t[:-1], abs_t[1:])[0,1][1,0] if T>1 else 0.0
        # burstiness
        nzz = np.flatnonzero(abs_t > 0)
        if len(nzz) < 2:
            burstiness = 0.0
        else:
            inter_arrival_times = np.diff(nzz)
            mean_iat = np.mean(inter_arrival_times)
            var_iat = np.var(inter_arrival_times)
            burstiness = (var_iat / mean_iat + Epsilon)

        l1 = np.sum(abs_t)
        l2 = np.sqrt(np.sum(abs_t ** 2))

        features = np.concatenate((mean_abs, std_abs, max_abs, p99_abs, p90_abs, nnz_rate, lag1))
        return features
    
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