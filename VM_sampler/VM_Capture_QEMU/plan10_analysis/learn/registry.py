#!/usr/bin/env python3
"""registry.py -- the Learn palette: tiers, shapes, modules, and what each needs.

A module declares the shapes it accepts and the shape it emits, its parameters, its kind (for a
model: classifier, clusterer, novelty, reconstructor, encoder), and the packages it needs. A
module whose package is missing is listed as unavailable with the reason; a designed model that
is not built yet is listed as unbuilt with what it would need. The page shows both by name and
the validator refuses them by name (the wavelet / scattering precedent). Nothing is hidden.

Shapes: rows (n, f); path (n, W, k), a sequence of block vectors; image (n, W, P), the page-by-
time tile; embedding (n, d), rows produced by a model; series tiles are read as paths with k = 1.
"""
from __future__ import annotations

import importlib.util

SHAPES = {
    "rows":      {"label": "rows",      "desc": "one vector per tile: the metrics a run wrote, or a flattened tile"},
    "path":      {"label": "path",      "desc": "frames x blocks per tile: a walk through block-space"},
    "image":     {"label": "image",     "desc": "frames x pages per tile: the page-by-time image"},
    "embedding": {"label": "embedding", "desc": "rows produced by an encoder"},
    "scores":    {"label": "scores",    "desc": "what a model says per tile"},
    "results":   {"label": "results",   "desc": "the scored run"},
}

TIERS = [
    {"k": "input",      "idx": "01", "name": "Input",      "color": "#78838f"},
    {"k": "preprocess", "idx": "02", "name": "Preprocess", "color": "#0f8ba3"},
    {"k": "represent",  "idx": "03", "name": "Represent",  "color": "#6d28d9"},
    {"k": "model",      "idx": "04", "name": "Model",      "color": "#d97706"},
    {"k": "score",      "idx": "05", "name": "Score",      "color": "#b8860b"},
    {"k": "output",     "idx": "06", "name": "Output",     "color": "#1c8a4e"},
]

DEP_LABEL = {"sklearn": "scikit-learn", "torch": "torch", "pyod": "pyod", "aeon": "aeon", "tabpfn": "tabpfn",
             "chronos": "chronos-forecasting", "momentfm": "momentfm", "timesfm": "timesfm", "torchvision": "torchvision",
             "umap": "umap-learn"}


def have(pkg: str) -> bool:
    try:
        return importlib.util.find_spec(pkg) is not None
    except (ImportError, ValueError):
        return False


def deps() -> dict:
    return {k: have(k) for k in DEP_LABEL}


def _num(k, lab, default, step=None, lo=None, hi=None):
    return {"k": k, "kind": "number", "lab": lab, "default": default, "step": step, "min": lo, "max": hi}


def _sel(k, lab, opts, default):
    return {"k": k, "kind": "select", "lab": lab, "opts": opts, "default": default}


def _bool(k, lab, default):
    return {"k": k, "kind": "bool", "lab": lab, "default": default}


def _mod(id, tier, name, ico, desc, accepts=(), emits=None, params=(), needs=(), kind=None, ref=None, unbuilt=None, flags=()):
    d = {"id": id, "tier": tier, "name": name, "ico": ico, "desc": desc, "accepts": list(accepts), "emits": emits,
         "params": list(params), "needs": list(needs), "kind": kind, "ref": ref, "flags": list(flags)}
    if unbuilt:
        d["unbuilt"] = unbuilt
    return d


TORCH_EPOCHS = _num("epochs", "epochs", 30, 1, 1, 2000)
TORCH_LR = _num("lr", "learning rate", 0.001, 0.0001, 1e-6, 1.0)
TORCH_BATCH = _num("batch", "batch size", 64, 1, 1, 4096)
HIDDEN = _num("hidden", "hidden width", 32, 1, 1, 4096)
BANK = _sel("bank", "bank", [["family", "one model per family, argmin error = predicted family (B1 phase 1)"],
                             ["workload", "one model per workload"],
                             ["none", "one model on the whole training set: novelty score only"]], "family")
K_MODE = _sel("k_mode", "k", [["families", "k = number of families in training (fixed in advance)"],
                              ["workloads", "k = number of workloads in training"],
                              ["number", "k = the number below"]], "families")


def build_registry() -> dict:
    D = deps()
    M = [
        # ---- 01 input
        _mod("input", "input", "Input", "I", "what a run wrote: its metric rows, or the tiles it saved",
             emits="rows",
             params=[{"k": "run", "kind": "run", "lab": "run", "default": ""},
                     _sel("source", "read", [["features", "features.npz: one row of metrics per tile"],
                                             ["tiles", "tiles.npz: the tiles as written (path, image, or series as a 1-block path)"]], "features")]),
        # ---- 02 preprocess
        _mod("scale", "preprocess", "Scale", "s", "fit on the training fold only: standard, robust, min-max or quantile",
             accepts=["rows", "embedding", "path", "image"], emits="same", needs=["sklearn"],
             params=[_sel("method", "method", [["standard", "standard (z-score)"], ["robust", "robust (median / IQR)"],
                                               ["minmax", "min-max to [0, 1]"], ["quantile", "quantile to normal"]], "standard")]),
        _mod("log1p", "preprocess", "log1p", "ln", "log(1 + x) per value; counts and heavy tails read better",
             accepts=["rows", "embedding", "path", "image"], emits="same"),
        _mod("diff", "preprocess", "Difference", "d", "first difference along frames: the delta as a derivative, one frame shorter",
             accepts=["path", "image"], emits="same"),
        _mod("per_recording_z", "preprocess", "Per-recording z", "z/r",
             "z-score every tile against its own recording's mean and std (no labels used). A scientific choice: it removes level, "
             "which is what the APF vs wAPF arms are about",
             accepts=["rows", "path", "image"], emits="same", flags=["level_removed"]),
        _mod("flatten", "preprocess", "Flatten", "[ ]", "a path or image becomes one long row, so tabular models can read it",
             accepts=["path", "image"], emits="rows"),
        _mod("pool_pages", "preprocess", "Pool pages", "P/n", "mean-pool the page axis into n bins (the pages stay in address order)",
             accepts=["image"], emits="image", params=[_num("bins", "bins", 32, 1, 1, 4096)]),
        _mod("pca", "preprocess", "PCA", "pca", "principal components, fit on the training fold", accepts=["rows", "embedding"], emits="rows",
             needs=["sklearn"], params=[_num("n_components", "components", 8, 1, 1, 4096)]),
        _mod("kpca", "preprocess", "Kernel PCA", "kpc", "kernel principal components (rbf), fit on the training fold", accepts=["rows", "embedding"], emits="rows",
             needs=["sklearn"], params=[_num("n_components", "components", 8, 1, 1, 4096), _num("gamma", "gamma (0 = 1/f)", 0, 0.001, 0, 100)]),
        _mod("ica", "preprocess", "ICA", "ica", "independent components (FastICA), fit on the training fold", accepts=["rows", "embedding"], emits="rows",
             needs=["sklearn"], params=[_num("n_components", "components", 8, 1, 1, 4096)]),
        _mod("rproj", "preprocess", "Random projection", "rp", "Gaussian random projection to d dimensions (seeded)", accepts=["rows", "embedding"], emits="rows",
             needs=["sklearn"], params=[_num("n_components", "dimensions", 16, 1, 1, 4096)]),
        _mod("select", "preprocess", "Select features", "sel", "keep the k columns with the most variance, or the most mutual information with the family",
             accepts=["rows", "embedding"], emits="rows", needs=["sklearn"],
             params=[_sel("method", "method", [["variance", "variance"], ["mutual_info", "mutual information with the family (uses training labels)"]], "variance"),
                     _num("k", "k columns", 8, 1, 1, 4096)]),
        # ---- 03 represent
        _mod("ae_mlp", "represent", "AE (MLP)", "ae", "a small autoencoder (sklearn MLPRegressor, B1's); its bottleneck is the embedding",
             accepts=["rows", "embedding"], emits="embedding", needs=["sklearn"], kind="encoder",
             params=[_num("bottleneck", "bottleneck", 3, 1, 1, 512), _num("max_iter", "max iterations", 300, 1, 1, 10000)]),
        _mod("vae", "represent", "VAE", "vae", "variational autoencoder over flattened tiles (torch); the mean of the latent is the embedding",
             accepts=["rows", "embedding", "path", "image"], emits="embedding", needs=["torch"], kind="encoder", ref="Kingma and Welling 2013",
             params=[_num("latent", "latent size", 8, 1, 1, 512), HIDDEN, _num("beta", "beta (KL weight)", 1.0, 0.1, 0, 100), TORCH_EPOCHS, TORCH_LR, TORCH_BATCH]),
        _mod("conv_ae", "represent", "Conv AE", "cae", "convolutional autoencoder over a path or image (torch); the code is the embedding",
             accepts=["path", "image"], emits="embedding", needs=["torch"], kind="encoder",
             params=[_num("latent", "latent size", 16, 1, 1, 512), _num("channels", "conv channels", 16, 1, 1, 512), TORCH_EPOCHS, TORCH_LR, TORCH_BATCH]),
        _mod("ts_foundation", "represent", "Time-series foundation model", "FM", "a pretrained encoder (Chronos, MOMENT, TimesFM) as a frozen embedding",
             accepts=["path"], emits="embedding", needs=["torch", "chronos"], kind="encoder", ref="Ansari et al. 2024; Goswami et al. 2024; Das et al. 2024",
             unbuilt="designed: needs a package with weights (chronos-forecasting, momentfm or timesfm); download size reported before the first run"),
        _mod("image_backbone", "represent", "Frozen image backbone", "IB", "an ImageNet backbone (ResNet-18 / ConvNeXt-T) as a frozen embedding of the tile as a grayscale image",
             accepts=["image"], emits="embedding", needs=["torch", "torchvision"], kind="encoder", ref="He et al. 2016; Liu et al. 2022",
             unbuilt="designed: needs torchvision and its weights"),
        # ---- 04 model: classifiers
        _mod("logreg", "model", "Logistic regression", "lr", "the linear floor every other number is compared to", accepts=["rows", "embedding"], needs=["sklearn"], kind="classifier",
             params=[_num("C", "C", 1.0, 0.1, 1e-4, 1e4)]),
        _mod("svm_linear", "model", "Linear SVM", "svm", "a linear support-vector classifier", accepts=["rows", "embedding"], needs=["sklearn"], kind="classifier",
             params=[_num("C", "C", 1.0, 0.1, 1e-4, 1e4)]),
        _mod("knn", "model", "kNN", "knn", "k nearest neighbours", accepts=["rows", "embedding"], needs=["sklearn"], kind="classifier",
             params=[_num("k", "k", 5, 1, 1, 500)]),
        _mod("rf", "model", "Random forest", "rf", "the supervised foil B1 phase 2 used", accepts=["rows", "embedding"], needs=["sklearn"], kind="classifier",
             params=[_num("n_estimators", "trees", 200, 10, 1, 5000), _num("max_depth", "max depth (0 = none)", 0, 1, 0, 200)]),
        _mod("extratrees", "model", "Extra trees", "et", "extremely randomised trees", accepts=["rows", "embedding"], needs=["sklearn"], kind="classifier",
             params=[_num("n_estimators", "trees", 200, 10, 1, 5000)]),
        _mod("hgb", "model", "Gradient boosting", "gb", "histogram gradient boosting (sklearn's)", accepts=["rows", "embedding"], needs=["sklearn"], kind="classifier",
             params=[_num("max_iter", "boosting rounds", 200, 10, 1, 5000), _num("learning_rate", "learning rate", 0.1, 0.01, 1e-3, 1)]),
        _mod("mlp", "model", "MLP", "mlp", "a multilayer perceptron classifier (sklearn)", accepts=["rows", "embedding"], needs=["sklearn"], kind="classifier",
             params=[HIDDEN, _num("max_iter", "max iterations", 300, 1, 1, 10000)]),
        _mod("minirocket", "model", "MiniRocket", "rkt", "random convolution kernels with fixed dilations and bias quantiles, then a ridge classifier: "
             "near state of the art on the UCR archive, minutes on CPU, numpy only",
             accepts=["path"], needs=[], kind="classifier", ref="Dempster, Schmidt and Webb 2021",
             params=[_num("n_kernels", "kernels (multiple of 84)", 840, 84, 84, 10000)]),
        _mod("lstm", "model", "LSTM", "lstm", "a recurrent classifier over the path (torch)", accepts=["path"], needs=["torch"], kind="classifier",
             params=[HIDDEN, _num("layers", "layers", 1, 1, 1, 6), TORCH_EPOCHS, TORCH_LR, TORCH_BATCH]),
        _mod("gru", "model", "GRU", "gru", "a gated recurrent classifier over the path (torch)", accepts=["path"], needs=["torch"], kind="classifier",
             params=[HIDDEN, _num("layers", "layers", 1, 1, 1, 6), TORCH_EPOCHS, TORCH_LR, TORCH_BATCH]),
        _mod("rnn", "model", "RNN", "rnn", "a plain recurrent classifier over the path (torch)", accepts=["path"], needs=["torch"], kind="classifier",
             params=[HIDDEN, _num("layers", "layers", 1, 1, 1, 6), TORCH_EPOCHS, TORCH_LR, TORCH_BATCH]),
        _mod("cnn1d", "model", "1-D CNN", "c1", "convolution over frames (torch)", accepts=["path"], needs=["torch"], kind="classifier",
             params=[_num("channels", "conv channels", 32, 1, 1, 1024), _num("kernel", "kernel size", 3, 1, 1, 64), TORCH_EPOCHS, TORCH_LR, TORCH_BATCH]),
        _mod("tcn", "model", "TCN", "tcn", "a temporal convolutional network: dilated causal convolutions with residuals (torch)",
             accepts=["path"], needs=["torch"], kind="classifier", ref="Bai, Kolter and Koltun 2018",
             params=[_num("channels", "conv channels", 32, 1, 1, 1024), _num("levels", "dilation levels", 3, 1, 1, 10), TORCH_EPOCHS, TORCH_LR, TORCH_BATCH]),
        _mod("inceptiontime", "model", "InceptionTime", "inc", "inception blocks over the path, a single network of the ensemble (torch)",
             accepts=["path"], needs=["torch"], kind="classifier", ref="Ismail Fawaz et al. 2020",
             params=[_num("channels", "filters per branch", 16, 1, 1, 512), _num("blocks", "inception blocks", 2, 1, 1, 8), TORCH_EPOCHS, TORCH_LR, TORCH_BATCH]),
        _mod("transformer", "model", "Transformer (patches)", "tr", "a transformer encoder over patches of the path, PatchTST-style (torch)",
             accepts=["path"], needs=["torch"], kind="classifier", ref="Nie et al. 2023",
             params=[_num("d_model", "d_model", 32, 1, 4, 1024), _num("heads", "heads", 4, 1, 1, 32), _num("layers", "layers", 2, 1, 1, 12),
                     _num("patch", "patch length (frames)", 2, 1, 1, 256), TORCH_EPOCHS, TORCH_LR, TORCH_BATCH]),
        _mod("cnn2d", "model", "2-D CNN", "c2", "a small ResNet-style convolutional classifier over the image (torch)", accepts=["image", "path"], needs=["torch"], kind="classifier",
             params=[_num("channels", "conv channels", 16, 1, 1, 512), TORCH_EPOCHS, TORCH_LR, TORCH_BATCH]),
        _mod("ft_transformer", "model", "FT-Transformer", "ftt", "attention over the metrics as tokens (torch)", accepts=["rows", "embedding"], needs=["torch"], kind="classifier",
             ref="Gorishniy et al. 2021", params=[_num("d_model", "d_model", 32, 1, 4, 1024), _num("heads", "heads", 4, 1, 1, 32), _num("layers", "layers", 2, 1, 1, 12), TORCH_EPOCHS, TORCH_LR, TORCH_BATCH]),
        _mod("tabpfn", "model", "TabPFN v2", "pfn", "a pretrained tabular foundation model, built for small n", accepts=["rows", "embedding"], needs=["torch", "tabpfn"], kind="classifier",
             ref="Hollmann et al. 2025", unbuilt="designed: needs the tabpfn package and its weights"),
        # ---- model: clusterers
        _mod("kmeans", "model", "k-means", "km", "do families emerge unsupervised (B1 phase 2)", accepts=["rows", "embedding"], needs=["sklearn"], kind="clusterer",
             params=[K_MODE, _num("k", "k when k = number", 4, 1, 2, 500)]),
        _mod("gmm", "model", "GMM", "gmm", "a Gaussian mixture (B1 phase 2)", accepts=["rows", "embedding"], needs=["sklearn"], kind="clusterer",
             params=[K_MODE, _num("k", "k when k = number", 4, 1, 2, 500)]),
        _mod("agglomerative", "model", "Hierarchical", "hc", "agglomerative clustering, Ward linkage (B1 phase 2)", accepts=["rows", "embedding"], needs=["sklearn"], kind="clusterer",
             params=[K_MODE, _num("k", "k when k = number", 4, 1, 2, 500)]),
        # ---- model: novelty
        _mod("isoforest", "model", "Isolation forest", "if", "novelty score: how easily a tile is isolated", accepts=["rows", "embedding"], needs=["sklearn"], kind="novelty",
             params=[_num("n_estimators", "trees", 200, 10, 1, 5000)]),
        _mod("lof", "model", "LOF", "lof", "local outlier factor as a novelty score", accepts=["rows", "embedding"], needs=["sklearn"], kind="novelty",
             params=[_num("k", "neighbours", 20, 1, 1, 500)]),
        _mod("ocsvm", "model", "One-class SVM", "oc", "a one-class support-vector novelty score", accepts=["rows", "embedding"], needs=["sklearn"], kind="novelty",
             params=[_num("nu", "nu", 0.1, 0.01, 0.001, 1)]),
        _mod("ecod", "model", "ECOD", "ecod", "empirical-CDF outlier score, parameter free (numpy)", accepts=["rows", "embedding"], needs=[], kind="novelty",
             ref="Li et al. 2022"),
        _mod("knn_dist", "model", "kNN distance", "kd", "novelty score: distance to the k-th nearest training tile", accepts=["rows", "embedding"], needs=["sklearn"], kind="novelty",
             params=[_num("k", "k", 5, 1, 1, 500)]),
        _mod("deep_svdd", "model", "Deep SVDD", "svdd", "a network pulling benign tiles to one centre; distance is the novelty score (torch)",
             accepts=["rows", "embedding", "path", "image"], needs=["torch"], kind="novelty", ref="Ruff et al. 2018",
             params=[HIDDEN, _num("out", "output size", 8, 1, 1, 512), TORCH_EPOCHS, TORCH_LR, TORCH_BATCH]),
        # ---- model: reconstructors (bank-able)
        _mod("ae_bank", "model", "AE bank", "aeb", "one autoencoder per family; a tile belongs where it reconstructs best (B1 phase 1, sklearn MLPRegressor)",
             accepts=["rows", "embedding"], needs=["sklearn"], kind="reconstructor",
             params=[BANK, _num("bottleneck", "bottleneck", 3, 1, 1, 512), _num("max_iter", "max iterations", 300, 1, 1, 10000), _num("min_train", "min training tiles per member", 8, 1, 1, 100000)]),
        _mod("vae_bank", "model", "VAE bank", "vb", "one variational autoencoder per family over flattened tiles (torch)",
             accepts=["rows", "embedding", "path", "image"], needs=["torch"], kind="reconstructor",
             params=[BANK, _num("latent", "latent size", 8, 1, 1, 512), HIDDEN, _num("beta", "beta (KL weight)", 1.0, 0.1, 0, 100), TORCH_EPOCHS, TORCH_LR, TORCH_BATCH,
                     _num("min_train", "min training tiles per member", 8, 1, 1, 100000)]),
        _mod("conv_ae_bank", "model", "Conv AE bank", "cab", "one convolutional autoencoder per family over paths or images (torch)",
             accepts=["path", "image"], needs=["torch"], kind="reconstructor",
             params=[BANK, _num("latent", "latent size", 16, 1, 1, 512), _num("channels", "conv channels", 16, 1, 1, 512), TORCH_EPOCHS, TORCH_LR, TORCH_BATCH,
                     _num("min_train", "min training tiles per member", 8, 1, 1, 100000)]),
        # ---- designed, unbuilt
        _mod("usad", "model", "USAD", "usad", "adversarially trained autoencoders for sequence anomalies", accepts=["path"], needs=["torch"], kind="novelty", ref="Audibert et al. 2020",
             unbuilt="designed: not built"),
        _mod("tranad", "model", "TranAD", "tad", "transformer anomaly detector for sequences", accepts=["path"], needs=["torch"], kind="novelty", ref="Tuli et al. 2022",
             unbuilt="designed: not built"),
        _mod("s4", "model", "S4 / Mamba", "ssm", "state-space sequence models", accepts=["path"], needs=["torch"], kind="classifier", ref="Gu et al. 2022; Gu and Dao 2023",
             unbuilt="designed: not built; Mamba wants CUDA kernels"),
        _mod("ts2vec", "represent", "TS2Vec", "t2v", "contrastive self-supervised sequence embeddings", accepts=["path"], emits="embedding", needs=["torch"], kind="encoder", ref="Yue et al. 2022",
             unbuilt="designed: not built"),
        _mod("mae", "represent", "MAE / ViT", "mae", "masked autoencoder pretraining over image patches, then a probe", accepts=["image"], emits="embedding", needs=["torch"], kind="encoder",
             ref="He et al. 2022", unbuilt="designed: not built"),
        _mod("diffusion", "represent", "Diffusion", "dif", "generative modeling of tiles", accepts=["path", "image"], emits="embedding", needs=["torch"], kind="encoder",
             unbuilt="deferred to its own study (JK, 2026-09)"),
        # ---- 05 score
        _mod("score", "score", "Score", "=", "every score the model's kind allows, the null, the bootstrap, and what explains the result",
             accepts=["scores"], emits="results",
             params=[_num("null_permutations", "null: test labels permuted against the predictions, times", 200, 10, 0, 10000),
                     _num("kernel_null_permutations", "target archetype: kernel-level permutation, no refit (archetype labels permuted across the kernels), times", 500, 50, 0, 10000),
                     _num("bootstrap", "bootstrap over recordings, resamples", 200, 10, 0, 10000),
                     _num("fpr", "recall at false-positive rate", 0.05, 0.01, 0.001, 0.5),
                     _num("calibration_bins", "calibration bins", 10, 1, 2, 100),
                     _bool("over_time", "keep the score along t_index per recording", True),
                     _bool("importance", "permutation importance (rows) / per-block importance (paths)", True),
                     _bool("saliency", "input-gradient saliency for torch models", True)]),
        # ---- 06 output
        _mod("write", "output", "Write", ">", "learn_results.json, predictions, embeddings, confusion, and the sidecar; without it the run is not valid",
             accepts=["results"], params=[_bool("keep_embeddings", "keep embeddings", True), _bool("keep_predictions", "keep per-tile predictions", True)]),
    ]
    for m in M:
        missing = [DEP_LABEL.get(d, d) for d in m["needs"] if not D.get(d)]
        m["available"] = not missing and not m.get("unbuilt")
        m["why"] = m.get("unbuilt") or (f"needs {', '.join(missing)}" if missing else "")
    return {"schema": "plan10.learn.modules.v1", "shapes": SHAPES, "tiers": TIERS, "modules": M, "deps": D, "dep_labels": DEP_LABEL}


def by_id(reg: dict | None = None) -> dict:
    reg = reg or build_registry()
    return {m["id"]: m for m in reg["modules"]}


if __name__ == "__main__":
    import json
    r = build_registry()
    print(json.dumps({"deps": r["deps"], "available": [m["id"] for m in r["modules"] if m["available"]],
                      "unavailable": {m["id"]: m["why"] for m in r["modules"] if not m["available"]}}, indent=1))
