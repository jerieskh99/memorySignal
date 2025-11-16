import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import sk_metrics



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

    @staticmethod
    def _calc_macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_true.ndim != 1 or y_pred.ndim != 1:
            raise ValueError("y_true and y_pred must be 1D arrays.")
        f1 = sk_metrics.f1_score(y_true, y_pred, average='macro')
        return f1

    # Area Under the Receiver Operating Characteristic (ROC) curve
    @staticmethod
    def _calc_AUC_ROC(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        if y_true.ndim != 1 or y_scores.ndim != 1:
            raise ValueError("y_true and y_scores must be 1D arrays.")
        auc_roc = sk_metrics.roc_auc_score(y_true, y_scores)
        return auc_roc
    
    # Area under the precision-recall curve
    @staticmethod
    def _calc_AUC_PR(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        if y_true.ndim != 1 or y_scores.ndim != 1:
            raise ValueError("y_true and y_scores must be 1D arrays.")
        auc_pr = sk_metrics.average_precision_score(y_true, y_scores)
        return auc_pr


    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray) -> dict:
        metrics = {}
        metrics['macro_f1'] = Simple1NN._calc_macro_f1_score(y_true, y_pred)
        metrics['AUC_ROC'] = Simple1NN._calc_AUC_ROC(y_true, y_scores)
        metrics['AUC_PR'] = Simple1NN._calc_AUC_PR(y_true, y_scores)
        return metrics

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

    # This function performs ablation based on the specified mode 
    # (magnitude_only, phase_only, no_neighbors, no_neighbors_magnitude, no_neighbors_phase)
    @staticmethod 
    def _execute_ablation(mode: str, data: dict[str, np.ndarray] = None): 
        if mode not in ["full", "no_phase", "no_magnitude", "no_neighbors"]:
            raise ValueError("Invalid ablation mode specified, choose from 'magnitude_only'," \
            " 'phase_only', 'no_neighbors', 'no_neighbors_magnitude', 'no_neighbors_phase'.")
        
        if data is None:
            raise ValueError("Data must be provided for ablation.")
        
        mag = data['X_mag_train']
        phase = data['X_phase_train']
        mag_neighbors = data['x_mag_train_neighbors']
        phase_neighbors = data['X_phase_train_neighbors']
        
        if mode == "full":
            return np.hstack((mag, phase, mag_neighbors, phase_neighbors))
        
        if mode == "no_phase":
            return np.hstack((mag, mag_neighbors))
        
        if mode == "no_magnitude":
            return np.hstack((phase, phase_neighbors))
        
        if mode == "no_neighbors":
            return np.hstack((mag, phase))
        
        # SHOULD NOT REACH HERE
        raise ValueError("Unhandled ablation mode.")

    def train(self, spatial = False, ablation_mode = None, ablation_type = None):
        if ablation_mode is None and ablation_type is None:
            ValueError("Ablation mode and type must be specified together or not at all.")

        if ablation_mode and spatial:
            raise ValueError("Ablation mode and spatial features cannot be used together.")

        if ablation_mode:
            X_train = self._execute_ablation(ablation_mode, data={
                'X_mag_train': self.X_mag_train,
                'X_phase_train': self.X_phase_train,
                'x_mag_train_neighbors': self.x_mag_train_neighbors,
                'X_phase_train_neighbors': self.X_phase_train_neighbors
            })

        if not spatial and not ablation_mode:
            X_train = np.hstack((self.X_mag_train, self.X_phase_train))
        else:
            X_train = np.hstack((self.X_mag_train, self.X_phase_train,
                                 self.x_mag_train_neighbors, self.X_phase_train_neighbors))

        y_train = np.asarray(self.y_labels_data_routines) # Labels are simply the indices of the training samples
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def validate(self, spatial=False, split_ratio=0.2, ablation_mode = None, ablation_type = None):
        if ablation_mode is None and ablation_type is None:
            ValueError("Ablation mode and type must be specified together or not at all.")
        
        if ablation_mode and spatial:
            raise ValueError("Ablation mode and spatial features cannot be used together.")

        if ablation_mode :
            X_train = self._execute_ablation(ablation_mode, data={
                'X_mag_train': self.X_mag_train,
                'X_phase_train': self.X_phase_train,
                'x_mag_train_neighbors': self.x_mag_train_neighbors,
                'X_phase_train_neighbors': self.X_phase_train_neighbors
            })

        # X_train = np.hstack((self.X_mag_train, self.X_phase_train)) if not spatial else np.hstack((
        #     self.X_mag_train, self.X_phase_train, self.x_mag_train_neighbors, self.X_phase_train_neighbors))

        if not spatial and not ablation_mode:
            X_train = np.hstack((self.X_mag_train, self.X_phase_train))
        else:
            X_train = np.hstack((self.X_mag_train, self.X_phase_train,
                                 self.x_mag_train_neighbors, self.X_phase_train_neighbors))
        
        if not self.is_trained:
            ValueError("Model must be trained before validation.")
        
        N = X_train.shape[0]
        split_index = int(N * ( 1- split_ratio))
        val_set = X_train[split_index:, :]
        y_true = np.asarray(self.y_labels_data_routines[split_index:, :])
        y_pred = self.model.predict(val_set)
        accuracy = np.mean(y_pred == y_true)
        return accuracy # between 0 and 1
    
    def predict(self, X_test_set: np.ndarray, spatial=False, ret_scores=False, 
                ablation_mode = None, ablation_type = None): 

        if ablation_mode is None and ablation_type is None:
            ValueError("Ablation mode and type must be specified together or not at all.")
        
        if ablation_mode and spatial:
            raise ValueError("Ablation mode and spatial features cannot be used together.")

        if ablation_mode:
            X_test_set = self._execute_ablation(ablation_mode, data={
                'X_mag_train': self.X_mag_train,
                'X_phase_train': self.X_phase_train,
                'x_mag_train_neighbors': self.x_mag_train_neighbors,
                'X_phase_train_neighbors': self.X_phase_train_neighbors
            })

        if not self.is_trained:
            ValueError("Model must be trained before prediction.")
            
        if spatial and not ablation_mode:
            if X_test_set.shape[1] != (self.X_mag_train.shape[1] + self.X_phase_train.shape[1] +
                                    self.x_mag_train_neighbors.shape[1] + self.X_phase_train_neighbors.shape[1]):
                raise ValueError("X_test_set shape does not match expected spatial feature dimensions.")
        else:
            if X_test_set.shape[1] != (self.X_mag_train.shape[1] + self.X_phase_train.shape[1]):
                raise ValueError("X_test_set shape does not match expected non-spatial feature dimensions.")
        
        y_pred = self.model.predict(X_test_set)

        if ret_scores:
            dist, _ = self.model.kneighbors(X_test_set, n_neighbors=1, return_distance=True) 
            y_scores = -dist.flatten()  # Invert distances to get similarity scores

            return y_pred, y_scores
        
        else:
            return y_pred
        
    def get_model_params(self) -> dict:
        params = {
            'n_neighbors': self.model.n_neighbors,
            'weights': self.model.weights,
            'metric': self.model.metric,
            'algorithm': self.model.algorithm
        }
        return params
        

        
