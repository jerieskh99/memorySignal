import numpy as np
from utils.randPageSelector import Selector
from simple1NN import Simple1NN

class BaselineXOR:
    """
    Minimal class: only responsible for loading pages and computing XORs.
    No feature extraction, no NN.
    """
    def __init__(self, pathBitmap: str, pathPrev: str, pathCurr: str, numPages: int):
        self.selector = Selector(pathBitmap, numPages)
        self.pathCurr = pathCurr
        self.pathPrev = pathPrev
        self.numPages = numPages

    def calculateXor(self, pathA: str, pathB: str) -> np.ndarray:
        """
        Compute XOR between two dumps (prev, curr).
        Returns a 2D array with shape (P, B) = (numPagesSelected, numBytesPerPage).
        """
        pathA = self.pathPrev if not pathA else pathA
        pathB = self.pathCurr if not pathB else pathB

        dataA = self.selector.loadPages(pathA)
        dataB = self.selector.loadPages(pathB)

        if dataA.shape != dataB.shape:
            raise ValueError(f"Shape mismatch: {dataA.shape} vs {dataB.shape}")

        dataA = dataA.astype(np.uint8, copy=False)
        dataB = dataB.astype(np.uint8, copy=False)

        xorResult = np.bitwise_xor(dataA, dataB)

        return xorResult
    
    
    
class OneNNBaselineXOR():
    """
    Wrapper:
    - Holds BaselineXOR to compute XORs.
    - Extracts features from a full XOR time series (T, B).
    - Trains/uses a Simple1NN on those features.
    """
    def __init__(self, pathBitmap: str, pathPrev: str, pathCurr: str, numPages: int):
        self.baselineXOR = BaselineXOR(pathBitmap, pathPrev, pathCurr, numPages)
        self.nn = Simple1NN(neighbors=1, weights="uniform")
        self.LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8) #Lookup Table for Hamming weight

    def calculateXor(self, pathA: str, pathB: str) -> np.ndarray:
        return self.baselineXOR.calculateXor(pathA, pathB)
    
    def extractFeatures(self, xorData: np.ndarray) -> np.ndarray:
        """
        xorData: 2D array of shape (T, B)
            T = number of time steps (pairs of dumps)
            B = bytes per 'sample' (all pages concatenated or however you define it)

        Returns: feature vector of shape (8,)
        """
        if xorData.ndim != 2:
            raise ValueError("Input data must be a 2D array with shape T, B (numSamples, numBytesPerSample)")
        
        T, B = xorData.shape # T: numSamples, B: numBytesPerSample

        # Per-sample Hamming counts using LUT
        hamming_t = self.LUT[xorData].sum(axis=1).astype(np.int32)  # shape: (T,)

        # Ratio of flipped bits per sample (normalize by total bits per sample)
        total_bits_per_sample = float(B * 8)
        change_ratio_t = hamming_t / total_bits_per_sample  # shape (T,)

        # --- Parity: even/odd number of flips per sample ---
        # 0 -> even flips, 1 -> odd flips
        parity_t = (hamming_t & 1).astype(np.uint8) # shape: (T,)
        # Fraction of samples with odd parity
        parity_ratio = float(parity_t.mean()) # scalar

        # Aggregate stats over time
        hamming_mean = float(hamming_t.mean())
        hamming_max  = float(hamming_t.max())
        hamming_std  = float(hamming_t.std())

        ratio_mean = float(change_ratio_t.mean())
        ratio_max  = float(change_ratio_t.max())
        ratio_std  = float(change_ratio_t.std())

        # --- Global byte entropy over XOR values (0..255) ---
        # Flatten all bytes and compute histogram
        flat = xorData.ravel()
        counts = np.bincount(flat, minlength=256).astype(np.float64)
        probs = counts / counts.sum()
        # Avoid log2(0) by masking zero probabilities
        nonzero = probs > 0
        byte_entropy = float(-np.sum(probs[nonzero] * np.log2(probs[nonzero])))

        features = np.array(
            [
                hamming_mean, hamming_max, hamming_std,
                ratio_mean,   ratio_max,   ratio_std,
                parity_ratio, byte_entropy,
            ],
            dtype=np.float32,
        )
        
        return features
    
    def extractFeatures(self, xorData: np.ndarray) -> np.ndarray:
        return self.baselineXOR.extractFeatures(xorData)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X: shape (N, 8) feature vectors (each from one full time-series run)
        y: shape (N,) labels
        """
        self.nn.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        X: shape (M, 8) feature vectors
        """
        return self.nn.predict(X)
    
    

