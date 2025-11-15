import numpy as np
from sklearn.neighbors import KNeighborsClassifier



class Simple1NN:
    def __init__(self, X_mag_train: np.ndarray, X_phase_train: np.ndarray, x_mag_train_neighbors: np.ndarray,
                  X_phase_train_neighbors: np.ndarray, y_labels_data_routines: np.ndarray, metric = 'cosine', alogirithm = 'brute'):
        # KNN metric choice:
        # We use metric='cosine' for scale-invariant comparison across mixed feature groups
        # (cosine compares direction, not magnitude). In scikit-learn, cosine forces a
        # brute-force neighbor search (KDTree/BallTree don’t support cosine). If we z-score
        # (or L2-normalize) features, Euclidean becomes comparable; on L2-normalized vectors,
        # cosine and Euclidean yield the same neighbor ranking.
        self.model = KNeighborsClassifier(n_neighbors=1, weights='uniform', metric=metric, algorithm=alogirithm)
        self.is_trained = False

        self.X_mag_train = X_mag_train
        self.X_phase_train = X_phase_train  
        self.x_mag_train_neighbors = x_mag_train_neighbors
        self.X_phase_train_neighbors = X_phase_train_neighbors

        self.y_labels_data_routines = y_labels_data_routines

        self.metric = metric
        self.algorithm = alogirithm

        self._initial_sanity_checks()


    def _initial_sanity_checks(self):
        # ---------- Sanity checks (shape / finiteness / alignment) ----------
        # 2D matrices
        assert isinstance(self.X_mag_train, np.ndarray) and self.X_mag_train.ndim == 2, "X_mag_train must be (N, Dm)"
        assert isinstance(self.X_phase_train, np.ndarray) and self.X_phase_train.ndim == 2, "X_phase_train must be (N, Dp)"
        assert isinstance(self.x_mag_train_neighbors, np.ndarray) and self.x_mag_train_neighbors.ndim == 2, "x_mag_train_neighbors must be (N, Dmn)"
        assert isinstance(self.X_phase_train_neighbors, np.ndarray) and self.X_phase_train_neighbors.ndim == 2, "X_phase_train_neighbors must be (N, Dpn)"

        # Same number of samples (rows)
        N = self.X_mag_train.shape[0]
        assert self.X_phase_train.shape[0] == N, "phase rows must match mag rows"
        assert self.x_mag_train_neighbors.shape[0] == N, "mag-neigh rows must match mag rows"
        assert self.X_phase_train_neighbors.shape[0] == N, "phase-neigh rows must match mag rows"

        # Labels: 1D, aligned length
        y = np.asarray(self.y_labels_data_routines)
        assert y.ndim == 1, "y_labels_data_routines must be 1D of length N"
        assert y.shape[0] == N, "labels length must match number of rows"

        # Finite values (no NaN/Inf)
        assert np.isfinite(self.X_mag_train).all(), "X_mag_train contains non-finite values"
        assert np.isfinite(self.X_phase_train).all(), "X_phase_train contains non-finite values"
        assert np.isfinite(self.x_mag_train_neighbors).all(), "x_mag_train_neighbors contains non-finite values"
        assert np.isfinite(self.X_phase_train_neighbors).all(), "X_phase_train_neighbors contains non-finite values"

        # Metric / algorithm compatibility (cosine requires brute in sklearn)
        if self.metric == 'cosine':
            assert self.alogirithm == 'brute', "metric='cosine' requires algorithm='brute' in sklearn KNN"


    def train(self, spatial = False):
        if not spatial:
            X_train = np.hstack((self.X_mag_train, self.X_phase_train))
        else:
            X_train = np.hstack((self.X_mag_train, self.X_phase_train,
                                 self.x_mag_train_neighbors, self.X_phase_train_neighbors))
        y_train = np.asarray(self.y_labels_data_routines) # Labels are simply the indices of the training samples
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def validate(self, spatial=False, split_ratio=0.2):
        X_train = np.hstack((self.X_mag_train, self.X_phase_train)) if not spatial else np.hstack((
            self.X_mag_train, self.X_phase_train, self.x_mag_train_neighbors, self.X_phase_train_neighbors))
        
        if not self.is_trained:
            ValueError("Model must be trained before validation.")
        
        N = X_train.shape[0]
        split_index = int(N * ( 1- split_ratio))
        val_set = X_train[split_index:, :]
        y_true = y_train = np.asarray(self.y_labels_data_routines[split_index:, :])
        y_pred = self.model.predict(val_set)
        accuracy = np.mean(y_pred == y_true)
        return accuracy # between 0 and 1
    
    def predict(self, X_test_set: np.ndarray, spatial=False):
        if not self.is_trained:
            ValueError("Model must be trained before prediction.")
            
            if spatial:
                if X_test_set.shape[1] != (self.X_mag_train.shape[1] + self.X_phase_train.shape[1] +
                                        self.x_mag_train_neighbors.shape[1] + self.X_phase_train_neighbors.shape[1]):
                    raise ValueError("X_test_set shape does not match expected spatial feature dimensions.")
            else:
                if X_test_set.shape[1] != (self.X_mag_train.shape[1] + self.X_phase_train.shape[1]):
                    raise ValueError("X_test_set shape does not match expected non-spatial feature dimensions.")
            
            y_pred = self.model.predict(X_test_set)
            return y_pred
        
