#!/usr/bin/env python3
"""preprocess.py -- fitted transforms, fit on the training fold only.

Every transform has fit(X, y, keys) and transform(X, keys). The executor fits on the training
rows of a fold and applies to both sides; nothing here ever sees test labels. Two transforms
use unlabeled information from every row they touch and say so: per_recording_z (each tile
against its own recording's statistics, so a held-out recording normalises itself) and the
quantile scaler (its map is fit on train, applied to test, like any scaler).

Shapes: rows (n, f); path (n, W, k); image (n, W, P). A per-column scaler on a path or image
treats the last axis as the columns (blocks or pages), over every frame of every tile.
"""
from __future__ import annotations

import numpy as np


class PreprocessError(ValueError):
    pass


class Transform:
    emits = "same"

    def __init__(self, params: dict, seed: int = 0):
        self.p = dict(params or {})
        self.seed = int(seed)

    def fit(self, X, y=None, keys=None):
        return self

    def transform(self, X, keys=None):
        return X

    def describe(self) -> dict:
        return {}


def _cols(X: np.ndarray) -> np.ndarray:
    """(n, ...) -> (rows, columns): rows for a 2-D input, every frame for a 3-D one."""
    return X.reshape(-1, X.shape[-1]) if X.ndim == 3 else X


class Scale(Transform):
    def fit(self, X, y=None, keys=None):
        from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler, StandardScaler
        m = self.p.get("method", "standard")
        C = _cols(X)
        if m == "standard":
            self.s = StandardScaler()
        elif m == "robust":
            self.s = RobustScaler()
        elif m == "minmax":
            self.s = MinMaxScaler()
        elif m == "quantile":
            self.s = QuantileTransformer(n_quantiles=int(max(2, min(1000, C.shape[0]))), output_distribution="normal", random_state=self.seed)
        else:
            raise PreprocessError(f"unknown scale method {m!r}")
        self.s.fit(C)
        return self

    def transform(self, X, keys=None):
        out = self.s.transform(_cols(X)).astype(np.float32)
        return out.reshape(X.shape) if X.ndim == 3 else out

    def describe(self):
        return {"method": self.p.get("method", "standard")}


class Log1p(Transform):
    """Symmetric: sign(x) * log(1 + |x|), so angles and negative deltas survive."""

    def transform(self, X, keys=None):
        return (np.sign(X) * np.log1p(np.abs(X))).astype(np.float32)


class Diff(Transform):
    def transform(self, X, keys=None):
        if X.ndim != 3 or X.shape[1] < 2:
            raise PreprocessError("Difference needs a path or image with at least two frames")
        return np.diff(X, axis=1).astype(np.float32)


class PerRecordingZ(Transform):
    """Each tile against its own recording: rows per column, tiles as a whole for paths and images."""

    def transform(self, X, keys=None):
        if keys is None or "recording" not in keys:
            raise PreprocessError("per_recording_z needs the recording key")
        out = np.empty_like(X, dtype=np.float32)
        rec = np.asarray(keys["recording"])
        for r in np.unique(rec):
            m = rec == r
            sub = X[m]
            if X.ndim == 2:
                mu, sd = sub.mean(axis=0), sub.std(axis=0)
            else:
                mu, sd = sub.mean(), sub.std()
            sd = np.where(sd > 0, sd, 1.0) if np.ndim(sd) else (sd if sd > 0 else 1.0)
            out[m] = (sub - mu) / sd
        return out


class Flatten(Transform):
    emits = "rows"

    def transform(self, X, keys=None):
        return X.reshape(X.shape[0], -1).astype(np.float32)


class PoolPages(Transform):
    emits = "image"

    def transform(self, X, keys=None):
        if X.ndim != 3:
            raise PreprocessError("Pool pages needs an image")
        bins = int(self.p.get("bins", 32))
        if bins < 1 or bins > X.shape[2]:
            bins = min(max(1, bins), X.shape[2])
        parts = np.array_split(np.arange(X.shape[2]), bins)
        return np.stack([X[:, :, idx].mean(axis=2) for idx in parts], axis=2).astype(np.float32)


class _SkRows(Transform):
    emits = "rows"

    def _make(self, n_features: int):
        raise NotImplementedError

    def fit(self, X, y=None, keys=None):
        if X.ndim != 2:
            raise PreprocessError(f"{type(self).__name__} needs rows; Flatten a path or image first")
        self.t = self._make(X.shape[1], y)
        self.t.fit(X, y) if y is not None else self.t.fit(X)
        return self

    def transform(self, X, keys=None):
        return np.asarray(self.t.transform(X), dtype=np.float32)


class PCA(_SkRows):
    def _make(self, f, y=None):
        from sklearn.decomposition import PCA as P
        self.k = int(min(int(self.p.get("n_components", 8)), f))
        return P(n_components=self.k, random_state=self.seed)

    def fit(self, X, y=None, keys=None):
        super().fit(X, None, keys)
        self.k = int(min(self.k, X.shape[0]))
        if self.t.n_components_ != self.k:
            from sklearn.decomposition import PCA as P
            self.t = P(n_components=self.k, random_state=self.seed).fit(X)
        return self

    def describe(self):
        return {"n_components": getattr(self.t, "n_components_", None),
                "explained_variance_ratio": [float(v) for v in getattr(self.t, "explained_variance_ratio_", [])]}


class KernelPCA(_SkRows):
    def _make(self, f, y=None):
        from sklearn.decomposition import KernelPCA as K
        g = float(self.p.get("gamma", 0) or 0) or None
        return K(n_components=int(min(int(self.p.get("n_components", 8)), f)), kernel="rbf", gamma=g, random_state=self.seed)


class ICA(_SkRows):
    def _make(self, f, y=None):
        from sklearn.decomposition import FastICA
        return FastICA(n_components=int(min(int(self.p.get("n_components", 8)), f)), random_state=self.seed, max_iter=1000)


class RandomProjection(_SkRows):
    def _make(self, f, y=None):
        from sklearn.random_projection import GaussianRandomProjection
        return GaussianRandomProjection(n_components=int(self.p.get("n_components", 16)), random_state=self.seed)


class Select(_SkRows):
    def _make(self, f, y=None):
        from sklearn.feature_selection import SelectKBest, mutual_info_classif
        k = int(min(int(self.p.get("k", 8)), f))
        if self.p.get("method", "variance") == "mutual_info":
            self.uses_labels = True
            return SelectKBest(score_func=lambda X, y: mutual_info_classif(X, y, random_state=self.seed), k=k)
        self.uses_labels = False
        return SelectKBest(score_func=lambda X, y=None: X.var(axis=0), k=k)

    def fit(self, X, y=None, keys=None):
        if X.ndim != 2:
            raise PreprocessError("Select needs rows; Flatten a path or image first")
        self.t = self._make(X.shape[1], y)
        self.t.fit(X, y if self.uses_labels else np.zeros(X.shape[0]))
        return self

    def describe(self):
        return {"kept": [int(i) for i in np.flatnonzero(self.t.get_support())]}


REGISTRY = {"scale": Scale, "log1p": Log1p, "diff": Diff, "per_recording_z": PerRecordingZ, "flatten": Flatten,
            "pool_pages": PoolPages, "pca": PCA, "kpca": KernelPCA, "ica": ICA, "rproj": RandomProjection, "select": Select}


def make(module_id: str, params: dict, seed: int = 0) -> Transform:
    if module_id not in REGISTRY:
        raise PreprocessError(f"no preprocessing module {module_id!r}")
    return REGISTRY[module_id](params, seed)


def out_shape(module_id: str, in_shape: str) -> str:
    cls = REGISTRY.get(module_id)
    if cls is None:
        raise PreprocessError(f"no preprocessing module {module_id!r}")
    return in_shape if cls.emits == "same" else cls.emits
