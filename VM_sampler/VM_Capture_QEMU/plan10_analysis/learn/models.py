#!/usr/bin/env python3
"""models.py -- one wrapper interface over sklearn, numpy and torch models.

    m = make(module_id, params, seed, context)     context: n_families / n_workloads in training
    m.fit(X, y, keys)                              X in the shape the module accepts; y = labels (str)
    m.predict(X)                                   labels (classifier, bank), cluster ids (clusterer)
    m.scores(X)                                    {"proba": (n, C), "novelty": (n,), "recon_error": (n, members), ...}
    m.embed(X)                                     (n, d) or None
    m.saliency(X)                                  |d score / d input| per tile for torch models, else None
    m.classes_                                     the classes a classifier or bank can name
    m.train_curve                                  [(epoch, loss)] for torch models
    m.kind                                         classifier | clusterer | novelty | reconstructor | encoder

Every model is seeded from the configuration's seed; torch runs on the CPU unless
PLAN10_TORCH_DEVICE says otherwise, so a result is repeatable on another machine.
Novelty scores are oriented so that higher means more novel.
"""
from __future__ import annotations

import math
import os
import warnings

import numpy as np


class ModelError(ValueError):
    pass


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Model:
    kind = "classifier"

    def __init__(self, params: dict, seed: int = 0, context: dict | None = None):
        self.p = dict(params or {})
        self.seed = int(seed)
        self.ctx = dict(context or {})
        self.classes_ = None
        self.train_curve = None

    def fit(self, X, y=None, keys=None):
        return self

    def predict(self, X):
        raise ModelError(f"{type(self).__name__} does not predict")

    def scores(self, X) -> dict:
        return {}

    def embed(self, X):
        return None

    def saliency(self, X):
        return None

    def describe(self) -> dict:
        return {}


def _rows(X):
    if X.ndim != 2:
        raise ModelError("this model reads rows; Flatten a path or image first")
    return X


# ---------------------------------------------------------------------------
# sklearn classifiers
# ---------------------------------------------------------------------------

class SkClassifier(Model):
    kind = "classifier"

    def _make(self):
        raise NotImplementedError

    def fit(self, X, y=None, keys=None):
        X = _rows(X)
        y = np.asarray(y).astype(str)
        self.classes_ = sorted(set(y.tolist()))
        if len(self.classes_) < 2:
            self.only = self.classes_[0]
            self.est = None
            return self
        self.only = None
        self.est = self._make()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.est.fit(X, y)
        self.classes_ = [str(c) for c in self.est.classes_]
        return self

    def predict(self, X):
        X = _rows(X)
        if self.est is None:
            return np.array([self.only] * X.shape[0])
        return np.asarray(self.est.predict(X)).astype(str)

    def scores(self, X):
        X = _rows(X)
        if self.est is None:
            return {"proba": np.ones((X.shape[0], 1), dtype=np.float32)}
        if hasattr(self.est, "predict_proba"):
            return {"proba": np.asarray(self.est.predict_proba(X), dtype=np.float32)}
        d = np.asarray(self.est.decision_function(X))
        if d.ndim == 1:
            d = np.stack([-d, d], axis=1)
        return {"proba": _softmax(d).astype(np.float32), "proba_note": "softmax of margins, not calibrated"}

    def importances(self):
        e = getattr(self, "est", None)
        if e is not None and hasattr(e, "feature_importances_"):
            return [float(v) for v in e.feature_importances_]
        return None


class LogReg(SkClassifier):
    def _make(self):
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=float(self.p.get("C", 1.0)), max_iter=2000, random_state=self.seed)


class LinearSVM(SkClassifier):
    def _make(self):
        from sklearn.svm import LinearSVC
        return LinearSVC(C=float(self.p.get("C", 1.0)), random_state=self.seed, max_iter=5000)


class KNN(SkClassifier):
    def _make(self):
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_neighbors=int(self.p.get("k", 5)))

    def fit(self, X, y=None, keys=None):
        k = int(self.p.get("k", 5))
        self.p["k"] = max(1, min(k, X.shape[0]))
        return super().fit(X, y, keys)


class RandomForest(SkClassifier):
    def _make(self):
        from sklearn.ensemble import RandomForestClassifier
        md = int(self.p.get("max_depth", 0) or 0) or None
        return RandomForestClassifier(n_estimators=int(self.p.get("n_estimators", 200)), max_depth=md, random_state=self.seed, n_jobs=1)


class ExtraTrees(SkClassifier):
    def _make(self):
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(n_estimators=int(self.p.get("n_estimators", 200)), random_state=self.seed, n_jobs=1)


class HGB(SkClassifier):
    def _make(self):
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=int(self.p.get("max_iter", 200)), learning_rate=float(self.p.get("learning_rate", 0.1)),
                                              random_state=self.seed)


class SkMLP(SkClassifier):
    def _make(self):
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(hidden_layer_sizes=(int(self.p.get("hidden", 32)),), max_iter=int(self.p.get("max_iter", 300)), random_state=self.seed)


# ---------------------------------------------------------------------------
# MiniRocket, compact reimplementation in numpy (Dempster, Schmidt and Webb 2021)
# ---------------------------------------------------------------------------

class MiniRocket(Model):
    """84 fixed kernels of length 9 (three weights of +2, six of -1, every placement of the
    three), dilations spread over what the tile length allows, biases drawn from the quantiles
    of each kernel's convolution output on one training tile, and the feature is the proportion
    of positive values (PPV). Multivariate: each kernel sums the convolutions of a seeded random
    subset of channels. Then a ridge classifier with built-in CV over its alpha, on standardised
    features. Compact: the reference uses per-dilation feature counts and a fixed 10k features;
    here the count is the parameter, spread evenly."""
    kind = "classifier"

    def _kernels(self):
        from itertools import combinations
        ks = []
        for pos in combinations(range(9), 3):
            w = -np.ones(9, dtype=np.float32)
            w[list(pos)] = 2.0
            ks.append(w)
        return np.stack(ks)

    def _conv(self, X, w, dil, chans):
        """X (n, W, k) -> (n, W) summed over the chosen channels, zero padded to keep the length."""
        n, W, k = X.shape
        pad = 4 * dil
        Xp = np.pad(X[:, :, chans], ((0, 0), (pad, pad), (0, 0)))
        out = np.zeros((n, W), dtype=np.float32)
        for j in range(9):
            out += w[j] * Xp[:, j * dil: j * dil + W, :].sum(axis=2)
        return out

    def fit(self, X, y=None, keys=None):
        if X.ndim != 3:
            raise ModelError("MiniRocket reads paths (n, W, k)")
        from sklearn.linear_model import RidgeClassifierCV
        from sklearn.preprocessing import StandardScaler
        rng = np.random.default_rng(self.seed)
        n, W, k = X.shape
        self.K = self._kernels()
        max_dil = max(1, (W - 1) // 8)
        n_dil = min(max_dil, 4)
        dils = sorted({int(round(v)) for v in np.geomspace(1, max_dil, n_dil)}) if max_dil > 1 else [1]
        want = int(self.p.get("n_kernels", 840))
        per = max(1, want // (84 * len(dils)))
        self.spec = []          # (kernel index, dilation, channels, biases)
        y = np.asarray(y).astype(str)
        self.classes_ = sorted(set(y.tolist()))
        for d in dils:
            for ki in range(84):
                nch = int(rng.integers(1, min(k, 9) + 1))
                chans = np.sort(rng.choice(k, size=nch, replace=False))
                ex = int(rng.integers(0, n))
                c = self._conv(X[ex:ex + 1], self.K[ki], d, chans)[0]
                qs = np.quantile(c, (np.arange(per) + 0.5) / per) if c.size else np.zeros(per)
                self.spec.append((ki, d, chans, qs.astype(np.float32)))
        F = self._features(X)
        self.scaler = StandardScaler().fit(F)
        if len(self.classes_) < 2:
            self.est = None
            return self
        self.est = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)).fit(self.scaler.transform(F), y)
        self.classes_ = [str(c) for c in self.est.classes_]
        return self

    def _features(self, X):
        cols = []
        for ki, d, chans, qs in self.spec:
            c = self._conv(X, self.K[ki], d, chans)
            for b in qs:
                cols.append((c > b).mean(axis=1))
        return np.stack(cols, axis=1).astype(np.float32)

    def embed(self, X):
        return self.scaler.transform(self._features(X)).astype(np.float32)

    def predict(self, X):
        if self.est is None:
            return np.array([self.classes_[0]] * X.shape[0])
        return np.asarray(self.est.predict(self.embed(X))).astype(str)

    def scores(self, X):
        if self.est is None:
            return {"proba": np.ones((X.shape[0], 1), dtype=np.float32)}
        d = np.asarray(self.est.decision_function(self.embed(X)))
        if d.ndim == 1:
            d = np.stack([-d, d], axis=1)
        return {"proba": _softmax(d).astype(np.float32), "proba_note": "softmax of ridge margins, not calibrated"}

    def describe(self):
        return {"n_features": len(self.spec) * (len(self.spec[0][3]) if self.spec else 0), "dilations": sorted({s[1] for s in self.spec}),
                "alpha": float(getattr(getattr(self, "est", None), "alpha_", float("nan")))}


# ---------------------------------------------------------------------------
# clusterers
# ---------------------------------------------------------------------------

class Clusterer(Model):
    kind = "clusterer"

    def _k(self, y):
        mode = self.p.get("k_mode", "families")
        if mode == "number":
            return max(2, int(self.p.get("k", 4)))
        if mode == "workloads":
            return max(2, int(self.ctx.get("n_workloads") or 2))
        return max(2, int(self.ctx.get("n_families") or 2))

    def fit(self, X, y=None, keys=None):
        X = _rows(X)
        self.k = min(self._k(y), max(2, X.shape[0]))
        self._fit(X)
        return self

    def describe(self):
        return {"k": int(self.k), "k_mode": self.p.get("k_mode", "families")}


class KMeans(Clusterer):
    def _fit(self, X):
        from sklearn.cluster import KMeans as KM
        self.est = KM(n_clusters=self.k, n_init=10, random_state=self.seed).fit(X)
        self.centroids = self.est.cluster_centers_

    def predict(self, X):
        return self.est.predict(_rows(X)).astype(int)


class GMM(Clusterer):
    def _fit(self, X):
        from sklearn.mixture import GaussianMixture
        self.est = GaussianMixture(n_components=self.k, random_state=self.seed, reg_covar=1e-4).fit(X)
        self.centroids = self.est.means_

    def predict(self, X):
        return self.est.predict(_rows(X)).astype(int)

    def scores(self, X):
        return {"proba": np.asarray(self.est.predict_proba(_rows(X)), dtype=np.float32)}


class Agglomerative(Clusterer):
    """Ward linkage on the training rows; test rows join the nearest training centroid, since
    agglomerative clustering has no predict of its own."""

    def _fit(self, X):
        from sklearn.cluster import AgglomerativeClustering
        lab = AgglomerativeClustering(n_clusters=self.k, linkage="ward").fit_predict(X)
        self.centroids = np.stack([X[lab == c].mean(axis=0) for c in range(self.k)])

    def predict(self, X):
        X = _rows(X)
        d = ((X[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)
        return d.argmin(axis=1).astype(int)


# ---------------------------------------------------------------------------
# novelty scorers (higher = more novel)
# ---------------------------------------------------------------------------

class Novelty(Model):
    kind = "novelty"

    def fit(self, X, y=None, keys=None):
        X = _rows(X)
        self._fit(X)
        return self

    def predict(self, X):
        raise ModelError("a novelty scorer has no labels to predict; read its novelty score")


class IsoForest(Novelty):
    def _fit(self, X):
        from sklearn.ensemble import IsolationForest
        self.est = IsolationForest(n_estimators=int(self.p.get("n_estimators", 200)), random_state=self.seed).fit(X)

    def scores(self, X):
        return {"novelty": (-self.est.score_samples(_rows(X))).astype(np.float32)}


class LOF(Novelty):
    def _fit(self, X):
        from sklearn.neighbors import LocalOutlierFactor
        k = max(1, min(int(self.p.get("k", 20)), X.shape[0] - 1))
        self.est = LocalOutlierFactor(n_neighbors=k, novelty=True).fit(X)

    def scores(self, X):
        return {"novelty": (-self.est.score_samples(_rows(X))).astype(np.float32)}


class OCSVM(Novelty):
    def _fit(self, X):
        from sklearn.svm import OneClassSVM
        self.est = OneClassSVM(nu=float(self.p.get("nu", 0.1)), gamma="scale").fit(X)

    def scores(self, X):
        return {"novelty": (-self.est.decision_function(_rows(X))).astype(np.float32)}


class ECOD(Novelty):
    """Empirical-CDF outlier detection (Li et al. 2022): per dimension, the left and right tail
    probabilities of a value under the training marginal; the score is the sum over dimensions
    of -log of the smaller tail (the paper's O_left / O_right / O_auto aggregation, here the
    skewness-free 'max of the two' form). Parameter free."""

    def _fit(self, X):
        self.train = np.sort(X, axis=0)
        self.n = X.shape[0]

    def scores(self, X):
        X = _rows(X)
        left = np.stack([np.searchsorted(self.train[:, j], X[:, j], side="right") for j in range(X.shape[1])], axis=1) / self.n
        right = 1.0 - np.stack([np.searchsorted(self.train[:, j], X[:, j], side="left") for j in range(X.shape[1])], axis=1) / self.n
        eps = 1.0 / (self.n + 1)
        tail = np.minimum(np.clip(left, eps, 1), np.clip(right, eps, 1))
        return {"novelty": (-np.log(tail)).sum(axis=1).astype(np.float32)}


class KNNDistance(Novelty):
    def _fit(self, X):
        from sklearn.neighbors import NearestNeighbors
        self.k = max(1, min(int(self.p.get("k", 5)), X.shape[0]))
        self.nn = NearestNeighbors(n_neighbors=self.k).fit(X)

    def scores(self, X):
        d, _ = self.nn.kneighbors(_rows(X))
        return {"novelty": d[:, -1].astype(np.float32)}


# ---------------------------------------------------------------------------
# reconstruction banks (B1 phase 1, generalised to any reconstructor)
# ---------------------------------------------------------------------------

class SkAE:
    """B1's autoencoder: MLPRegressor hidden -> bottleneck -> out, tanh, adam."""

    def __init__(self, bottleneck, max_iter, seed):
        from sklearn.neural_network import MLPRegressor
        self.est = MLPRegressor(hidden_layer_sizes=(int(bottleneck),), activation="tanh", solver="adam", max_iter=int(max_iter),
                                random_state=seed, tol=1e-5)

    def fit(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.est.fit(X, X)
        return self

    def error(self, X):
        Xh = self.est.predict(X)
        if Xh.ndim == 1:
            Xh = Xh.reshape(X.shape)
        return ((X - Xh) ** 2).mean(axis=1)

    def embed(self, X):
        return np.tanh(X @ self.est.coefs_[0] + self.est.intercepts_[0]).astype(np.float32)


class Bank(Model):
    """One reconstructor per member of `bank` (family or workload) in the training rows; a
    tile is predicted to belong where it reconstructs best, and its novelty is the smallest
    error. bank=none fits one reconstructor on everything: novelty only. A member with fewer
    than min_train tiles is not fitted and is named in `describe`."""
    kind = "reconstructor"

    def __init__(self, params, seed=0, context=None, member=None):
        super().__init__(params, seed, context)
        self.member_factory = member

    def fit(self, X, y=None, keys=None):
        by = self.p.get("bank", "family")
        min_train = int(self.p.get("min_train", 8))
        if by == "none" or y is None:
            groups = {"all": np.arange(X.shape[0])}
        else:
            lab = np.asarray(keys[by] if by == "workload" and keys is not None else y).astype(str)
            groups = {g: np.flatnonzero(lab == g) for g in sorted(set(lab.tolist()))}
        self.members, self.skipped = {}, {}
        for g, idx in groups.items():
            if idx.size < min_train:
                self.skipped[g] = int(idx.size)
                continue
            self.members[g] = self.member_factory(self.seed).fit(X[idx])
        if not self.members:
            raise ModelError(f"no bank member has {min_train} training tiles (sizes {dict(self.skipped)})")
        self.classes_ = sorted(self.members)
        return self

    def _errors(self, X):
        return np.stack([self.members[g].error(X) for g in self.classes_], axis=1).astype(np.float32)

    def predict(self, X):
        E = self._errors(X)
        return np.asarray(self.classes_)[E.argmin(axis=1)]

    def scores(self, X):
        E = self._errors(X)
        out = {"recon_error": E, "members": list(self.classes_), "novelty": E.min(axis=1)}
        if len(self.classes_) > 1:
            out["proba"] = _softmax(-E / (E.mean() + 1e-12)).astype(np.float32)
            out["proba_note"] = "softmax of negative reconstruction error over the bank, not calibrated"
        return out

    def embed(self, X):
        m = self.members[self.classes_[0]]
        return m.embed(X) if hasattr(m, "embed") else None

    def describe(self):
        return {"bank": self.p.get("bank", "family"), "members": list(self.classes_), "skipped_too_small": self.skipped}


def _ae_bank(params, seed, context):
    return Bank(params, seed, context, member=lambda s: SkAE(params.get("bottleneck", 3), params.get("max_iter", 300), s))


# ---------------------------------------------------------------------------
# torch: shared training loop, then the networks
# ---------------------------------------------------------------------------

def _torch():
    try:
        import torch
    except ImportError as e:
        raise ModelError("needs torch") from e
    return torch


def _device():
    torch = _torch()
    name = os.environ.get("PLAN10_TORCH_DEVICE", "cpu")
    return torch.device(name)


def _seed_all(seed):
    torch = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))


class TorchNet(Model):
    """A network with forward(x) -> (logits or output, features). Subclasses build `net`
    from the input shape and say what the loss is."""

    def __init__(self, params, seed=0, context=None):
        super().__init__(params, seed, context)
        self.epochs = int(self.p.get("epochs", 30))
        self.lr = float(self.p.get("lr", 1e-3))
        self.batch = int(self.p.get("batch", 64))

    def _build(self, shape, n_out):
        raise NotImplementedError

    def _tensor(self, X):
        torch = _torch()
        return torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.dev)

    def _train(self, X, target=None, loss_fn=None):
        torch = _torch()
        _seed_all(self.seed)
        self.dev = _device()
        n = X.shape[0]
        Xt = self._tensor(X)
        Tt = None if target is None else torch.as_tensor(target, device=self.dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.train_curve = []
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        for ep in range(self.epochs):
            self.net.train()
            perm = torch.randperm(n, generator=gen)
            tot = 0.0
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch].to(self.dev)
                xb = Xt[idx]
                out, feat = self.net(xb)
                loss = loss_fn(out, feat, xb, None if Tt is None else Tt[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss.item()) * idx.numel()
            self.train_curve.append((ep + 1, tot / max(1, n)))
        self.net.eval()

    def _forward_np(self, X):
        torch = _torch()
        self.net.eval()
        outs, feats = [], []
        with torch.no_grad():
            for i in range(0, X.shape[0], 512):
                o, f = self.net(self._tensor(X[i:i + 512]))
                outs.append(o.cpu().numpy())
                feats.append(f.cpu().numpy())
        return np.concatenate(outs), np.concatenate(feats)

    def embed(self, X):
        return self._forward_np(X)[1].astype(np.float32)

    def saliency(self, X):
        """|d(max output) / d input| per tile, in the input's shape."""
        torch = _torch()
        self.net.eval()
        out = []
        for i in range(0, X.shape[0], 256):
            xb = self._tensor(X[i:i + 256]).requires_grad_(True)
            o, _ = self.net(xb)
            s = o.max(dim=1).values.sum() if o.ndim == 2 else o.sum()
            g, = torch.autograd.grad(s, xb)
            out.append(g.abs().detach().cpu().numpy())
        return np.concatenate(out).astype(np.float32)


def _mlp(torch, d_in, hidden, d_out):
    nn = torch.nn
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, d_out))


class TorchClassifier(TorchNet):
    kind = "classifier"

    def fit(self, X, y=None, keys=None):
        torch = _torch()
        y = np.asarray(y).astype(str)
        self.classes_ = sorted(set(y.tolist()))
        if len(self.classes_) < 2:
            self.net = None
            return self
        yi = np.array([self.classes_.index(v) for v in y], dtype=np.int64)
        _seed_all(self.seed)
        self.dev = _device()
        self.net = self._build(X.shape[1:], len(self.classes_)).to(self.dev)
        ce = torch.nn.CrossEntropyLoss()
        self._train(X, yi, lambda out, feat, xb, t: ce(out, t))
        return self

    def predict(self, X):
        if self.net is None:
            return np.array([self.classes_[0]] * X.shape[0])
        return np.asarray(self.classes_)[self._forward_np(X)[0].argmax(axis=1)]

    def scores(self, X):
        if self.net is None:
            return {"proba": np.ones((X.shape[0], 1), dtype=np.float32)}
        return {"proba": _softmax(self._forward_np(X)[0]).astype(np.float32)}

    def embed(self, X):
        return None if self.net is None else super().embed(X)

    def saliency(self, X):
        return None if self.net is None else super().saliency(X)


class _Head:
    """features (n, d) -> logits; kept separate so every net returns (logits, features)."""


def _flat(shape):
    return int(np.prod(shape))


class TorchMLPClassifier(TorchClassifier):
    def _build(self, shape, n_out):
        torch = _torch()
        nn = torch.nn
        h = int(self.p.get("hidden", 32))

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.body = nn.Sequential(nn.Flatten(), nn.Linear(_flat(shape), h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
                s.head = nn.Linear(h, n_out)

            def forward(s, x):
                f = s.body(x)
                return s.head(f), f
        return Net()


class Recurrent(TorchClassifier):
    cell = "LSTM"

    def _build(self, shape, n_out):
        torch = _torch()
        nn = torch.nn
        W, k = shape
        h, layers = int(self.p.get("hidden", 32)), int(self.p.get("layers", 1))
        cell = {"LSTM": nn.LSTM, "GRU": nn.GRU, "RNN": nn.RNN}[self.cell]

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.rnn = cell(k, h, num_layers=layers, batch_first=True)
                s.head = nn.Linear(h, n_out)

            def forward(s, x):
                o, _ = s.rnn(x)
                f = o[:, -1, :]
                return s.head(f), f
        return Net()


class LSTMClassifier(Recurrent):
    cell = "LSTM"


class GRUClassifier(Recurrent):
    cell = "GRU"


class RNNClassifier(Recurrent):
    cell = "RNN"


class CNN1D(TorchClassifier):
    def _build(self, shape, n_out):
        torch = _torch()
        nn = torch.nn
        W, k = shape
        c, ks = int(self.p.get("channels", 32)), int(self.p.get("kernel", 3))

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.body = nn.Sequential(nn.Conv1d(k, c, ks, padding="same"), nn.ReLU(), nn.Conv1d(c, c, ks, padding="same"), nn.ReLU(),
                                       nn.AdaptiveAvgPool1d(1), nn.Flatten())
                s.head = nn.Linear(c, n_out)

            def forward(s, x):
                f = s.body(x.transpose(1, 2))
                return s.head(f), f
        return Net()


class TCN(TorchClassifier):
    """Dilated causal convolutions with residual connections (Bai, Kolter and Koltun 2018)."""

    def _build(self, shape, n_out):
        torch = _torch()
        nn = torch.nn
        W, k = shape
        c, levels = int(self.p.get("channels", 32)), int(self.p.get("levels", 3))

        class Block(nn.Module):
            def __init__(s, cin, cout, dil):
                super().__init__()
                s.pad = 2 * dil
                s.c1 = nn.Conv1d(cin, cout, 3, dilation=dil)
                s.c2 = nn.Conv1d(cout, cout, 3, dilation=dil)
                s.down = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
                s.act = nn.ReLU()

            def forward(s, x):
                y = s.act(s.c1(nn.functional.pad(x, (s.pad, 0))))
                y = s.act(s.c2(nn.functional.pad(y, (s.pad, 0))))
                return s.act(y + s.down(x))

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                blocks, cin = [], k
                for i in range(levels):
                    blocks.append(Block(cin, c, 2 ** i))
                    cin = c
                s.body = nn.Sequential(*blocks)
                s.head = nn.Linear(c, n_out)

            def forward(s, x):
                f = s.body(x.transpose(1, 2))[:, :, -1]
                return s.head(f), f
        return Net()


class InceptionTime(TorchClassifier):
    """One network of the InceptionTime ensemble (Ismail Fawaz et al. 2020): a bottleneck, three
    parallel convolutions of widths 10 / 20 / 40 (clipped to the tile), a max-pool branch, a
    residual every third block, global average pooling."""

    def _build(self, shape, n_out):
        torch = _torch()
        nn = torch.nn
        W, k = shape
        c, nb = int(self.p.get("channels", 16)), int(self.p.get("blocks", 2))
        widths = [max(1, min(w, W)) for w in (10, 20, 40)]
        widths = [w - 1 if w % 2 == 0 and w > 1 else w for w in widths]      # odd widths: 'same' padding without a copy

        class Inc(nn.Module):
            def __init__(s, cin):
                super().__init__()
                s.bott = nn.Conv1d(cin, c, 1) if cin > 1 else nn.Identity()
                cb = c if cin > 1 else cin
                s.convs = nn.ModuleList([nn.Conv1d(cb, c, w, padding="same") for w in widths])
                s.pool = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(cin, c, 1))
                s.bn = nn.BatchNorm1d(4 * c)
                s.act = nn.ReLU()

            def forward(s, x):
                z = s.bott(x)
                return s.act(s.bn(torch.cat([cv(z) for cv in s.convs] + [s.pool(x)], dim=1)))

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.blocks = nn.ModuleList()
                cin = k
                for i in range(nb):
                    s.blocks.append(Inc(cin))
                    cin = 4 * c
                s.res = nn.Conv1d(k, 4 * c, 1)
                s.head = nn.Linear(4 * c, n_out)

            def forward(s, x):
                x = x.transpose(1, 2)
                z = x
                for b in s.blocks:
                    z = b(z)
                z = torch.relu(z + s.res(x))
                f = z.mean(dim=2)
                return s.head(f), f
        return Net()


class PatchTransformer(TorchClassifier):
    """A transformer encoder over patches of frames (PatchTST-style, Nie et al. 2023), mean pooled."""

    def _build(self, shape, n_out):
        torch = _torch()
        nn = torch.nn
        W, k = shape
        d, heads, layers = int(self.p.get("d_model", 32)), int(self.p.get("heads", 4)), int(self.p.get("layers", 2))
        d = max(heads, (d // heads) * heads)
        pl = max(1, min(int(self.p.get("patch", 2)), W))
        npatch = W // pl

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.proj = nn.Linear(pl * k, d)
                s.pos = nn.Parameter(torch.zeros(1, npatch, d))
                layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=0.0, batch_first=True)
                s.enc = nn.TransformerEncoder(layer, layers)
                s.head = nn.Linear(d, n_out)

            def forward(s, x):
                n = x.shape[0]
                x = x[:, :npatch * pl, :].reshape(n, npatch, pl * k)
                z = s.enc(s.proj(x) + s.pos)
                f = z.mean(dim=1)
                return s.head(f), f
        return Net()


class CNN2D(TorchClassifier):
    """A small ResNet-style classifier over the tile as one-channel image (frames x pages or blocks)."""

    def _build(self, shape, n_out):
        torch = _torch()
        nn = torch.nn
        c = int(self.p.get("channels", 16))

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.c1 = nn.Sequential(nn.Conv2d(1, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU())
                s.c2 = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU())
                s.c3 = nn.Sequential(nn.Conv2d(c, 2 * c, 3, padding=1), nn.BatchNorm2d(2 * c), nn.ReLU())
                s.pool = nn.AdaptiveAvgPool2d(1)
                s.head = nn.Linear(2 * c, n_out)

            def forward(s, x):
                z = s.c1(x[:, None, :, :])
                z = s.c2(z) + z
                f = s.pool(s.c3(z)).flatten(1)
                return s.head(f), f
        return Net()


class FTTransformer(TorchClassifier):
    """Feature tokens (one linear embedding per metric) with a class token through a transformer encoder."""

    def _build(self, shape, n_out):
        torch = _torch()
        nn = torch.nn
        f_in = _flat(shape)
        d, heads, layers = int(self.p.get("d_model", 32)), int(self.p.get("heads", 4)), int(self.p.get("layers", 2))
        d = max(heads, (d // heads) * heads)

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.w = nn.Parameter(torch.randn(f_in, d) * 0.02)
                s.b = nn.Parameter(torch.zeros(f_in, d))
                s.cls = nn.Parameter(torch.zeros(1, 1, d))
                layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=0.0, batch_first=True)
                s.enc = nn.TransformerEncoder(layer, layers)
                s.head = nn.Linear(d, n_out)

            def forward(s, x):
                x = x.flatten(1)
                tok = x[:, :, None] * s.w[None] + s.b[None]
                z = s.enc(torch.cat([s.cls.expand(x.shape[0], -1, -1), tok], dim=1))
                f = z[:, 0]
                return s.head(f), f
        return Net()


class DeepSVDD(TorchNet):
    """One-class: a network pulling every training tile to one centre (the mean of the initial
    outputs); the novelty score is the squared distance to it (Ruff et al. 2018)."""
    kind = "novelty"

    def fit(self, X, y=None, keys=None):
        torch = _torch()
        _seed_all(self.seed)
        self.dev = _device()
        h, o = int(self.p.get("hidden", 32)), int(self.p.get("out", 8))
        body = _mlp(torch, _flat(X.shape[1:]), h, o)

        class Net(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.body = body

            def forward(s, x):
                f = s.body(x.flatten(1))
                return f, f
        self.net = Net().to(self.dev)
        with torch.no_grad():
            self.c = self.net(self._tensor(X))[0].mean(dim=0)
            self.c[self.c.abs() < 1e-3] = 1e-3
        self._train(X, None, lambda out, feat, xb, t: ((out - self.c) ** 2).sum(dim=1).mean())
        return self

    def scores(self, X):
        o, _ = self._forward_np(X)
        return {"novelty": ((o - self.c.cpu().numpy()) ** 2).sum(axis=1).astype(np.float32)}


class TorchVAE:
    """Dense VAE over flattened tiles: a member for banks, an encoder for the represent tier."""

    def __init__(self, params, seed):
        self.p, self.seed = dict(params), int(seed)

    def fit(self, X):
        torch = _torch()
        _seed_all(self.seed)
        self.dev = _device()
        self.shape = X.shape[1:]
        d_in, lat, h = _flat(self.shape), int(self.p.get("latent", 8)), int(self.p.get("hidden", 32))
        beta = float(self.p.get("beta", 1.0))
        nn = torch.nn

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.enc = nn.Sequential(nn.Linear(d_in, h), nn.ReLU())
                s.mu, s.lv = nn.Linear(h, lat), nn.Linear(h, lat)
                s.dec = nn.Sequential(nn.Linear(lat, h), nn.ReLU(), nn.Linear(h, d_in))

            def forward(s, x):
                e = s.enc(x.flatten(1))
                mu, lv = s.mu(e), s.lv(e).clamp(-8, 8)
                z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if s.training else mu
                return s.dec(z), mu, lv
        self.net = Net().to(self.dev)
        n = X.shape[0]
        Xt = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=float(self.p.get("lr", 1e-3)))
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        bs = int(self.p.get("batch", 64))
        self.train_curve = []
        for ep in range(int(self.p.get("epochs", 30))):
            self.net.train()
            perm = torch.randperm(n, generator=gen)
            tot = 0.0
            for i in range(0, n, bs):
                idx = perm[i:i + bs].to(self.dev)
                xb = Xt[idx]
                rec, mu, lv = self.net(xb)
                loss = ((rec - xb.flatten(1)) ** 2).mean(dim=1).mean() + beta * (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(dim=1)).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss.item()) * idx.numel()
            self.train_curve.append((ep + 1, tot / max(1, n)))
        self.net.eval()
        return self

    def _run(self, X):
        torch = _torch()
        self.net.eval()
        recs, mus = [], []
        with torch.no_grad():
            for i in range(0, X.shape[0], 512):
                xb = torch.as_tensor(np.asarray(X[i:i + 512], dtype=np.float32), device=self.dev)
                rec, mu, _ = self.net(xb)
                recs.append(((rec - xb.flatten(1)) ** 2).mean(dim=1).cpu().numpy())
                mus.append(mu.cpu().numpy())
        return np.concatenate(recs), np.concatenate(mus)

    def error(self, X):
        return self._run(X)[0]

    def embed(self, X):
        return self._run(X)[1].astype(np.float32)


class TorchConvAE:
    """Convolutional autoencoder: Conv1d over frames for a path (channels = blocks), Conv2d over
    (frames, pages) for an image; the code after adaptive pooling is the embedding."""

    def __init__(self, params, seed, shape_kind):
        self.p, self.seed, self.kind_ = dict(params), int(seed), shape_kind

    def fit(self, X):
        torch = _torch()
        _seed_all(self.seed)
        self.dev = _device()
        nn = torch.nn
        lat, c = int(self.p.get("latent", 16)), int(self.p.get("channels", 16))
        W = X.shape[1]
        if self.kind_ == "path":
            k = X.shape[2]

            class Net(nn.Module):
                def __init__(s):
                    super().__init__()
                    s.enc = nn.Sequential(nn.Conv1d(k, c, 3, padding="same"), nn.ReLU(), nn.Conv1d(c, c, 3, padding="same"), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(c, lat))
                    s.dec = nn.Sequential(nn.Linear(lat, c * W), nn.ReLU(), nn.Unflatten(1, (c, W)), nn.Conv1d(c, c, 3, padding="same"), nn.ReLU(), nn.Conv1d(c, k, 3, padding="same"))

                def forward(s, x):
                    z = s.enc(x.transpose(1, 2))
                    return s.dec(z).transpose(1, 2), z
        else:
            P = X.shape[2]
            Wp, Pp = max(1, W // 2), max(1, P // 2)

            class Net(nn.Module):
                def __init__(s):
                    super().__init__()
                    s.enc = nn.Sequential(nn.Conv2d(1, c, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, ceil_mode=True), nn.Conv2d(c, c, 3, padding=1), nn.ReLU(),
                                          nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, lat))
                    s.dec = nn.Sequential(nn.Linear(lat, c * Wp * Pp), nn.ReLU(), nn.Unflatten(1, (c, Wp, Pp)), nn.Upsample(size=(W, P)),
                                          nn.Conv2d(c, c, 3, padding=1), nn.ReLU(), nn.Conv2d(c, 1, 3, padding=1))

                def forward(s, x):
                    z = s.enc(x[:, None])
                    return s.dec(z)[:, 0], z
        self.net = Net().to(self.dev)
        n = X.shape[0]
        Xt = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=float(self.p.get("lr", 1e-3)))
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        bs = int(self.p.get("batch", 64))
        self.train_curve = []
        for ep in range(int(self.p.get("epochs", 30))):
            self.net.train()
            perm = torch.randperm(n, generator=gen)
            tot = 0.0
            for i in range(0, n, bs):
                idx = perm[i:i + bs].to(self.dev)
                xb = Xt[idx]
                rec, _ = self.net(xb)
                loss = ((rec - xb) ** 2).flatten(1).mean(dim=1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss.item()) * idx.numel()
            self.train_curve.append((ep + 1, tot / max(1, n)))
        self.net.eval()
        return self

    def _run(self, X):
        torch = _torch()
        self.net.eval()
        errs, zs = [], []
        with torch.no_grad():
            for i in range(0, X.shape[0], 256):
                xb = torch.as_tensor(np.asarray(X[i:i + 256], dtype=np.float32), device=self.dev)
                rec, z = self.net(xb)
                errs.append(((rec - xb) ** 2).flatten(1).mean(dim=1).cpu().numpy())
                zs.append(z.cpu().numpy())
        return np.concatenate(errs), np.concatenate(zs)

    def error(self, X):
        return self._run(X)[0]

    def embed(self, X):
        return self._run(X)[1].astype(np.float32)


class Encoder(Model):
    """The represent tier: fit an autoencoder on the training rows, emit its code."""
    kind = "encoder"

    def __init__(self, params, seed=0, context=None, member=None):
        super().__init__(params, seed, context)
        self.member_factory = member

    def fit(self, X, y=None, keys=None):
        self.m = self.member_factory(self.seed).fit(X)
        self.train_curve = getattr(self.m, "train_curve", None)
        return self

    def embed(self, X):
        return self.m.embed(X)

    def scores(self, X):
        return {"recon_error": self.m.error(X)[:, None], "members": ["all"], "novelty": self.m.error(X)}

    def transform(self, X):
        return self.embed(X)


# ---------------------------------------------------------------------------
# the factory
# ---------------------------------------------------------------------------

def make(module_id: str, params: dict, seed: int = 0, context: dict | None = None, shape: str = "rows") -> Model:
    p, ctx = dict(params or {}), dict(context or {})
    simple = {"logreg": LogReg, "svm_linear": LinearSVM, "knn": KNN, "rf": RandomForest, "extratrees": ExtraTrees, "hgb": HGB, "mlp": SkMLP,
              "minirocket": MiniRocket, "kmeans": KMeans, "gmm": GMM, "agglomerative": Agglomerative,
              "isoforest": IsoForest, "lof": LOF, "ocsvm": OCSVM, "ecod": ECOD, "knn_dist": KNNDistance,
              "lstm": LSTMClassifier, "gru": GRUClassifier, "rnn": RNNClassifier, "cnn1d": CNN1D, "tcn": TCN,
              "inceptiontime": InceptionTime, "transformer": PatchTransformer, "cnn2d": CNN2D, "ft_transformer": FTTransformer,
              "deep_svdd": DeepSVDD}
    if module_id in simple:
        return simple[module_id](p, seed, ctx)
    if module_id == "ae_bank":
        return _ae_bank(p, seed, ctx)
    if module_id == "vae_bank":
        return Bank(p, seed, ctx, member=lambda s: TorchVAE(p, s))
    if module_id == "conv_ae_bank":
        return Bank(p, seed, ctx, member=lambda s: TorchConvAE(p, s, "path" if shape == "path" else "image"))
    if module_id == "ae_mlp":
        return Encoder(p, seed, ctx, member=lambda s: SkAE(p.get("bottleneck", 3), p.get("max_iter", 300), s))
    if module_id == "vae":
        return Encoder(p, seed, ctx, member=lambda s: TorchVAE(p, s))
    if module_id == "conv_ae":
        return Encoder(p, seed, ctx, member=lambda s: TorchConvAE(p, s, "path" if shape == "path" else "image"))
    raise ModelError(f"no model {module_id!r} is built")
