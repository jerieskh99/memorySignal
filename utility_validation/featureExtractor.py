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
        X0 = X[: -1]  # X_t
        X1 = X[1:]  # X_{t+1}
        mu = X.mean(axis=0, keepdims=True) # (1, B)
        num = ((X0 - mu) * (X1 - mu)).sum(axis=0) # (B, )
        den = ((X - mu) ** 2).sum(axis=0) + epsilon # (B, )
        return num / den # (B, )

    @staticmethod
    def _extract_percentile(X: np.ndarray, p: int):
        return np.percentile(X, p, axis=0) # (B, )

    # @staticmethod
    # def _extract_burstiness(X: np.ndarray, epsilon=1e-8):
    #     nzz = np.flatnonzero(X > 0) # Indices of non-zero entries
    #     if len(nzz) < 2:
    #         burstiness = 0.0
    #     else:
    #         inter_arrival_times = np.diff(nzz) # Differences between consecutive indices
    #         mean_iat = np.mean(inter_arrival_times) # Mean inter-arrival time
    #         var_iat = np.var(inter_arrival_times)   # Variance of inter-arrival times
    #         burstiness = (var_iat / mean_iat + epsilon) / (var_iat / mean_iat + 1 + epsilon)
    #     return burstiness # (B, )
    
    @staticmethod
    def _extract_burstiness(X: np.ndarray, epsilon=1e-8):
        T, B = X.shape
        out = np.zeros(B, dtype=X.dtype) # (B, ) burstiness per block
        active_times = X > epsilon # (T, B)

        for b in range(B):
            indices = np.flatnonzero(active_times[:, b]) # Indices of non-zero entries for block b
            if indices.size >= 2:
                inter_arrival_times = np.diff(indices) # Differences between consecutive indices
                out[b] = inter_arrival_times.var() / (inter_arrival_times.mean() + epsilon) # Burstiness calculation
            else:
                out[b] = 0.0
        return out # (B, )

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
    
    # def _extract_flips_rate(self, X: np.ndarray):
    #     T, B = X.shape()
    #     flips = np.zeros(B)
    #     nz = X > 0

    #     if T < 2:
    #         return flips

    #     flips = (nz[1:] != nz[:-1]).mean(axis=0)

    def _extract_flips_rate(self, X: np.ndarray):
        # X: (T,B) boolean-ish activity → flips per column (B,)
        T, B = X.shape() 
        if T < 2: # not enough time steps
            return np.zeros(B) # (B, )
        
        nz = X > self.epsilon # (T, B) boolean
        return (nz[1:] != nz[:-1]).mean(axis=0) # (B, ) - fraction of time steps with flips

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
        lag1 = self._lag1_autocorr(abs_t, Epsilon) # (B, )

        # burstiness
        burstiness = self._extract_burstiness(abs_t) # (B, )

        # Gini index
        gini_index = self._extract_gini_index(abs_t) # (B, )

        # Flips in sign rate over time
        flips = self._extract_flips_rate(abs_t) # (B, )

        # L1 and L2 norms
        l1 = np.sum(abs_t, axis=0) # (B, )
        l2 = np.sqrt(np.sum(abs_t ** 2, axis=0)) # (B, )

        features = np.stack([mean_abs, std_abs, max_abs, p99_abs, p90_abs, 
                             nnz_rate, sparsity_abs, lag1, burstiness, gini_index, 
                             flips, l1, l2], axis=1) # (B, 13)
        names = [
            "mag_mean","mag_std","mag_max","mag_p99","mag_p90",
            "mag_nnz_rate","mag_sparsity","mag_lag1","mag_burstiness","mag_gini", 
            "mag_switch_rate","mag_l1","mag_l2"
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
        # Implementation:r4
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
                diff_angles = np.angle(np.exp(1j * phase_t[1:][:-1])) # (T-1,B)
                # 2) Compute mean absolute difference of these angles:
                weights = diff_angles.astype(float) / (np.sum(diff_angles, axis=0, keepdims=True)) # (T-1,B)
            else:
                weights = np.full((T-1, B), 1.0 / max(T-1, 1.0)) # (T-1,B)
            
            consistency_adjacent = np.cos(diff_angles * weights).sum(axis=0) # (B, )
        else:
            consistency_adjacent = np.zeros(B)
    
        # consistency_adjacent = np.mean(np.abs(np.diff(phase_t, axis=0)), axis=0) if T>1 else 0.0

        features = np.stack([c, s, R, phase_var, consistency_adjacent], axis=1) # (B, 5)

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
        offsets = np.array(list(range(-K, 0)) + list(range(1, K + 1))) # [-k,...,-1, 1,...,k] # shape [2k, ]
        idx = np.arange(N)[:None] + offsets[None:]  # shape [N, 2k]
        if mode == "cyclic":
            idx = idx % N
        elif mode == "clip":
            idx = np.clip(idx, 0, N-1)
        else:
            ValueError(f"Unknown mode: {mode}. Use only \"cyclic\" or \"clip\".")
        return idx  # shape [N, 2k]
    
    @staticmethod
    def entropy_mag(X: np.ndarray, epsilon=1e-8, axis=1, base=None, keepdims=False) -> np.ndarray:
        """
        Compute the entropy of the magnitude profiles along the specified axis.
        X: np.ndarray
        epsilon: small constant to avoid log(0)
        axis: axis along which to compute entropy
        base: logarithm base (None for natural log) Use base=2 for bits (ln(x) / ln(2) = log2(x))
        keepdims: whether to keep the reduced dimensions ([N,1,T] instead of [N,T])
        """
        S = np.sum(X, axis=axis, keepdims=True) + epsilon # ∑ m over time
        XlogX = np.where(X>0, X * np.log(X), 0.0) # m * log(m)
        numenator = XlogX.sum(axis=axis, keepdims=True)  # ∑ m * log(m)

        # H = log(Σ m) - (Σ m log m)/(Σ m)
        H = np.where(S>0, np.log(S) - numenator / S, 0.0)

        if base is not None:
            H = H / np.log(base)
        
        if not keepdims:
            H = np.squeeze(H, axis=axis)
        
        return H
    
    @staticmethod
    # def _extract_neighbors_gini_index(X_neighs: np.ndarray, epsilon = 1e-8, squeezeExtraDims = True) -> np.ndarray:
    #     T, K, B = X_neighs.shape
    #     gini_indices = np.zeros(B)
    #     for j in range(B):
    #         sorted_X = np.sort((X_neighs[:, :, j]).ravel())  # Sort over time and neighbors
    #         sumBlock = sorted_X.sum() # Sum over time and neighbors
    #         if sumBlock <= 0:
    #             gini_indices[j] = 0.0
    #             continue
        
    #         i = np.arange(1, T + 1, dtype=sorted_X.dtype)
    #         gini_indices[j] = ((2 * i - T - 1) @ sorted_X) / (T * sumBlock)
    #     return gini_indices # (B, )

    def _extract_neighbors_gini_index(X_neighs: np.ndarray) -> np.ndarray:
        """
        Compute Gini inequality over neighbor magnitudes for each focal block,
        returning TWO shape-consistent views:

        Input
        -----
        X_neighs : np.ndarray of shape (N, K, T)
            Neighbor tensor per focal block:
            - N = number of focal blocks
            - K = number of neighbors per block
            - T = number of time steps
            Each slice X_neighs[n] is the (K, T) matrix of neighbor values for block n.

        Output
        ------
        gini_pooled : np.ndarray of shape (N,)
            ONE scalar per block. For block n, we flatten its neighbor–time matrix
            X_neighs[n] to length M = K*T, sort it, and apply the standard Gini
            formula with M (not T!) as the length:
                G = ((2*i - M - 1) · x_sorted) / (M * sum(x)),  where i = 1..M
            Use this if you want a single inequality summary per block and do not
            need time resolution.

        gini_timewise : np.ndarray of shape (N, T)
            ONE scalar per block per time. For each time t, we compute Gini across
            the K neighbors (length K), preserving temporal structure:
                G_t = ((2*i - K - 1) · x_sorted_t) / (K * sum_t), where i = 1..K
            Use this when you want neighbor inequality as a time series to feed
            temporal models or to summarize later (e.g., time mean/median).

        Notes on correctness (dimensions)
        ---------------------------------
        • DO NOT use T in the pooled formula: pooled length is M = K*T.
        • For time-wise Gini, the length is K (neighbors) at each time step.
        • Zeros/empty cases are handled by returning 0.0 where the sum is ≤ 0.

        """
        N, K, T = X_neighs.shape
        gini_timewise = np.zeros((N, T), dtype=X_neighs.dtype)  # (N, T)
        gini_pooled   = np.zeros(N, dtype=float) # (N, )

        i = np.arange(1, K + 1, dtype=float) # (K, ) (neighbors) for time-wise Gini

        for j in range(N):  # for each block
            x_neighs = X_neighs[N] # (K, T) neighbors for block j

            # ---- time-wise across neighbors → (T,)
            sorted_X = np.sort((x_neighs), axis=0)  # (K, T), sorted over neighbors per time
            sumBlock = sorted_X.sum(axis=0) # Sum over neighbors per time (T, )     # (T, )
            if sumBlock <= 0:
                gini_timewise[j] = 0.0
            else:
                gini_timewise[j] = ((2 * i - K - 1) @ sorted_X) / (K * sumBlock)

            # ---- pooled over neighbors×time → scalar
            x_flat = x_neighs.ravel()  # (K*T, )
            s = x_flat.sum()         # scalar
            if s<=0:
                gini_pooled[j] = 0.0
            else:
                x_flat_sorted = np.sort(x_flat)  # (K*T, )
                M = x_flat_sorted.size
                indices_m = np.arange(1, M+1, dtype=float)
                gini_pooled[j] = ((2 * indices_m - M - 1) @ x_flat_sorted) / (M * s)

        return gini_pooled, gini_timewise  # (N, T)
    
    @staticmethod
    def _extract_hoyer_sparsity(X_neighs: np.ndarray, l1: np.ndarray, l2: np.ndarray, epsilon = 1e-8, squeezeExtraDims = True) -> np.ndarray:
        _, K, _ = X_neighs.shape
        sqrt_K = np.sqrt(K)
        hoyer_sparsity = (sqrt_K - (l1 / (l2 + epsilon))) / (sqrt_K - 1 + epsilon)
        return hoyer_sparsity  # (B, ) # shape matcnhes l1, l2 shapes  (e.g., (N,T))



    @staticmethod
    def calc_neighbors_mag_features(X: np.ndarray, neighbor_indices: np.ndarray, epsilon = 1e-8, squeezeExtraDims = True, entropyBase = 2) -> np.ndarray:
        """
        abs_all:   [N, T]  per-block magnitude time series  (|Δ| for each block over time)
        neigh_idx: [N, 2k] neighbor indices from build_neighbor_indices()

        returns:
        neigh_abs_mean: [N, T]  mean |Δ| over neighbors for each block/time
        """
        neighbor_abs = X[neighbor_indices, :]  # shape [N, 2k, T]
        neigh_abs_mean = np.mean(neighbor_abs, axis=1)  # shape [N, T]
        neigh_abs_std = np.std(neighbor_abs, axis=1)  # shape [N, T]
        neigh_abs_var = np.var(neighbor_abs, axis=1)  # shape [N, T]

        # Energy and RMS:
        neigh_abs_energy = np.sum(neighbor_abs ** 2, axis=1)  # shape [N, T]
        neigh_abs_RMS = np.sqrt(neigh_abs_energy / neighbor_abs.shape[1])  # shape [N, T]

        # Norms:
        neigh_abs_l1 = np.sum(neighbor_abs, axis=1) # shape [N, T]
        neigh_abs_l2 = np.sqrt(np.sum(neighbor_abs**2, axis=1)) # shape [N, T]
        neigh_abs_l_inf = np.max(neighbor_abs, axis=1) # shape [N, T]

        # Coeff of variation:
        neigh_abs_CV = neigh_abs_std / (neigh_abs_mean + epsilon) # shape [N, T]

        # Mean Absolute Deviation (over the neighbors):
        neigh_abs_MAD = np.mean(np.abs(neighbor_abs - neigh_abs_mean[:, None, :]), axis=1) # shape [N, T]

        # Entroy of magnitude profile (shape) [N, T] 
        keepdims = False if squeezeExtraDims else True
        H_m = FeatureExtractor.entropy_mag(neighbor_abs, epsilon=epsilon, axis=1, base=entropyBase, keepdims=keepdims)

        # Gini index over neighbors
        gini_pooled, gini_timewise = FeatureExtractor._extract_neighbors_gini_index(neighbor_abs)  # shape [N, T]

        # Hoyer sparsity over neighbors
        neigh_abs_hoyer = FeatureExtractor._extract_hoyer_sparsity(neighbor_abs, neigh_abs_l1, neigh_abs_l2, epsilon=epsilon, squeezeExtraDims=squeezeExtraDims)  # shape [N, T]

        # Peak to Average Ratio (PAR):
        par = neigh_abs_l_inf / (neigh_abs_mean + epsilon)  # shape [N, T]

        features = np.stack([neigh_abs_mean, neigh_abs_std, neigh_abs_var, neigh_abs_energy, neigh_abs_RMS, neigh_abs_l1, neigh_abs_l2, neigh_abs_l_inf,
                             neigh_abs_CV, neigh_abs_MAD, H_m, gini_pooled, gini_timewise, neigh_abs_hoyer, par], axis=1) # shape [N, 14, T]
        
        names = [
            "neigh_mag_mean","neigh_mag_std","neigh_mag_var","neigh_mag_energy","neigh_mag_RMS",
            "neigh_mag_l1","neigh_mag_l2","neigh_mag_linf","neigh_mag_CV","neigh_mag_MAD","neigh_mag_entropy",
            "neigh_mag_gini_timewise", "neigh_abs_hoyer", "neigh_mag_PAR"
        ]

        return {
            "names": names, 
            "values": features, # (N, 14, T)
            "pooled": {
                "names": ["neigh_mag_gini_pooled"],
                "values": gini_pooled          # (N,)
            }
        }
    
    @staticmethod
    def _extract_PPC_PairwiseAlignment(neighbor_abs: np.ndarray, R: np.ndarray, epsilon=1e-8) -> np.ndarray:
        _, K, T = neighbor_abs.shape
        ppc = (K * (R ** 2) - 1.0) / (K - 1.0 + epsilon) # shape [N, T]

        return ppc
    
    @staticmethod
    def _extract_second_trigonometric_moments(neighbor_abs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        second_moment = np.mean(np.exp(1j * 2 * neighbor_abs), axis=1) # shape [N, T]
        ro_2 = np.abs(second_moment) # shape [N, T]
        mu_2 = np.angle(second_moment) # shape [N, T]

        return second_moment, ro_2, mu_2
    
    @staticmethod
    def _extract_circular_MAD(neighbor_abs: np.ndarray, mu_N: np.ndarray, epsilon=1e-8) -> np.ndarray:
        """
        Circular Mean Absolute Deviation (MAD) around mean angle mu_N
        neighbor_abs: [N, 2k, T]
        mu_N:         [N, T]
        returns:
        circular_MAD: [N, T]
        """
        diff = neighbor_abs - mu_N[:, None, :]  # shape [N, 2k, T]
        wrapped = np.arctan2(np.sin(diff), np.cos(diff))  # shape [N, 2k, T]
        cMad = np.mean(np.abs(wrapped), axis=1)  # shape [N, T]
        return cMad
    
    @staticmethod
    def _extract_circular_entropy_fast(theta_neighbors: np.ndarray,
                          B: int = 36,
                          eps: float = 1e-12,
                          base: float | None = None) -> np.ndarray:
        """
        Fast circular (binned) entropy H_theta for neighbor phases.

        Parameters
        ----------
        theta_neighbors : np.ndarray
            Phases in radians with shape [N, K, T]
            N = focal blocks, K = neighbors, T = time steps.
        B : int
            Number of angular bins over (-pi, pi]. (e.g., 36 -> 10° bins)
        eps : float
            Small constant to avoid log(0) and division by 0.
        base : float | None
            Logarithm base. Use 2 for bits, 10 for log10, None for natural log (nats).

        Returns
        -------
        H : np.ndarray
            Entropy per block and time step, shape [N, T].
        """
        N, K, T = theta_neighbors.shape
        H = np.zeros((N, T), dtype = theta_neighbors.dtype)

        # Build B equal-width bins over (-pi, pi]
        bin_edges = np.linspace(-np.pi, np.pi, B+1, endpoint=True)  # B+1 edges for B bins

        # Loop over blocks and time steps
        for n in range(N):
            for t in range(T):
                phases = theta_neighbors[n, :, t]  # shape [K, ]
                counts, _ = np.histogram(phases, bins=bin_edges)
                counts = counts.astype(theta_neighbors.dtype)

                non_zero = counts > 0
                probabilities = (np.sum(counts[non_zero] * np.log(counts[non_zero] + eps)))/ (K + eps)  # sum of m * log(m)
                H_curr = np.log(K + eps) - probabilities  # H = log(Σ m) - (Σ m log m)/(Σ m)

                if base is not None:
                    H_curr = H_curr / np.log(base)
                
                H[n, t] = H_curr

        return H

    @staticmethod
    def calc_neighbors_phase_features(X: np.ndarray, neighbor_indices: np.ndarray, epsilon = 1e-8, squeezeExtraDims = True, entropyBase = 2) -> np.ndarray:
        """
        phi_all:   [N, T]  per-block phase/orientation time series (in radians)
        neigh_idx: [N, 2k]

        returns:
        mu_N:  [N, T]  circular mean of neighbors' phase at each time
        R_N:   [N, T]  order parameter (alignment strength) of neighbors at each time, in [0,1]
        """
        neighbor_abs = X[neighbor_indices, :]  # shape [N, 2k, T]
        N, K, T = neighbor_abs.shape
        neighbor_cos_mean = np.mean(np.cos(neighbor_abs), axis=1)  # shape [N, T]
        neighbor_sin_mean = np.mean(np.sin(neighbor_abs), axis=1)  # shape [N, T]
        mu_N = np.arctan2(neighbor_sin_mean, neighbor_cos_mean)  # shape [N, T]
        R_n = np.sqrt(neighbor_cos_mean**2 + neighbor_sin_mean**2)  # shape [N, T]
        C_v = 1 - R_n  # shape [N, T]
        csd = np.sqrt(-2 * np.log(R_n + epsilon))  # shape [N, T]
        ppc = FeatureExtractor._extract_PPC_PairwiseAlignment(neighbor_abs, R_n, epsilon=epsilon, squeezeExtraDims=squeezeExtraDims)
        m2, ro2, mu2 = FeatureExtractor._extract_second_trigonometric_moments(neighbor_abs, epsilon=epsilon, squeezeExtraDims=squeezeExtraDims)
        circular_skewness = ro2 * np.sin(mu2 - 2*mu_N)  # shape [N, T]
        kurtoisis = ro2 * np.cos(mu2 - 2*mu_N) - (R_n ** 4) # shape [N, T]
        rayleigh_statistic = K * R_n ** 2 # shape [N, T]
        phase_alignment_to_focal_block = np.mean(np.cos(neighbor_abs - X[:, None, :]), axis=1)  # shape [N, T]

        # Circular MAD (robust spread around mean angle):
        circular_MAD = FeatureExtractor._extract_circular_MAD(neighbor_abs, mu_N, epsilon=epsilon)  # shape [N, T]

        # Circular entropy (binned) 
        circular_entropy = FeatureExtractor._extract_circular_entropy_fast(neighbor_abs, B=36, eps=epsilon, base=entropyBase)  # shape [N, T]

        features = np.stack([mu_N, R_n, C_v, csd, ppc, m2, ro2, mu2, circular_skewness, kurtoisis, 
                             rayleigh_statistic, phase_alignment_to_focal_block, circular_MAD, circular_entropy], axis=1) # shape [N, 14, T]
        
        names = [
            "neigh_phase_circular_mean","neigh_phase_order_param_R","neigh_phase_circular_variance",
            "neigh_phase_circular_stddev","neigh_phase_PPC","neigh_phase_2nd_trig_moment","neigh_phase_2nd_trig_moment_Ro",
            "neigh_phase_2nd_trig_moment_mu", "neigh_phase_circular_skewness","neigh_phase_circular_kurtosis",
            "neigh_phase_rayleigh_statistic", "neigh_phase_alignment_to_focal", "neigh_phase_circular_MAD", "neigh_phase_circular_entropy"
        ]

        return {"names": names, "values": features}
    
    def calc_neighbors_features(self, X: np.ndarray, K: int, mode: str="cyclic", phase: bool=True, mag: bool=True):
        # X: (T, N) → transpose for neighbor fns that expect (N,T)
        if not mag and not phase:
            ValueError("At least one of mag or phase must be True")

        # X: np.ndarray  # (T, N)
        N = X.shape[1]
        neighbor_indices = self._extract_neighbors_indices(N, K, mode)
        X_t = X.T  # (N,T)

        mag_neigh = self.calc_neighbors_mag_features(X_t, neighbor_indices) if mag else None
        phase_neigh = self.calc_neighbors_phase_features(X_t, neighbor_indices) if phase else None

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