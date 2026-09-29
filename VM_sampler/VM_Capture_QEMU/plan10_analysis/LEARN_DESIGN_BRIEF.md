# Design brief: the Learn view

Written 2026-09-17 from a conversation with JK. **This is the design.** It was built the same
night; what exists, and how it departs from this text (a chain of slots rather than free-form
piping; the pooled null), is in README.md under "Learn". Read this for the why. It is the input for the prompts that will build it (envoy), in a later
session. Everything it cites about the existing code was read, not assumed; everything it
proposes is marked as a proposal.

## What it is, in one paragraph

A fourth full-screen view of the analysis console, sibling of Monitor and Explore: **Learn**.
Compose a learning pipeline as a graph of typed modules (input, preprocessing, model, score),
the way the scheme canvas composes extraction; point it at what runs have written; run it over
the honest splits with every fitted step inside the fold; sweep the combinations you choose;
get scores that go deeper than one accuracy number; see the results through the same five views
Explore already draws, plus the few that learning needs (confusion, ROC, embeddings, saliency).
Every configuration leaves a sidecar, the same discipline as a scheme run.

## What exists that it stands on

- **Runs write** `features.npz` (`X` tiles x metrics, `feature_names`, `tile_keys` with
  `recording, workload, family, block, t_index, seq_start`) plus `features.csv`, `sidecar.json`
  ([runner/executor.py](runner/executor.py) `_write`). Nothing else is written today.
- **In-process signals** ([runner/stages.py](runner/stages.py)): a Field is one row per changed
  page per pair; `block(field, wp, hp)` cuts the address axis into blocks; `collapse` reduces the
  page axis per pair and returns one Series per block for a blocked field; `window` makes Tiles
  `(n, W)` or `(n, W, k)` keyed by `(block, t_index, seq_start)`; `dense_page_matrix` makes the
  page-resolution tile, frames x pages, under a byte budget (256 MB default).
- **B1's modeling discipline** ([plan08_b1/b1_ae.py](../plan08_b1/b1_ae.py),
  [b1_splits.py](../plan08_b1/b1_splits.py)): per-fold `StandardScaler` fit on train only; folds
  from `fold_within_trace` (sanity ceiling), `fold_loro` (leave one rep out), `fold_lowo` (leave
  one workload out, the honest headline), grouped so no cell or workload straddles a split; a
  family with too few train windows is not modeled and its test windows fall to the others; fixed
  hyperparameters, one recorded seed, the one knob (bottleneck) not swept because effective n is
  workloads. The number that matters in B1 is the difference between arms (APF vs wAPF) on the
  same folds, not model vs model.
- **Explore** ([results_view.py](results_view.py), the `xp` overlay in the template): five views
  (distribution, over time, matrix, scatter, table) over one metric x one grouping key x one
  stat, numbers computed by numpy on the bridge, several runs aligned by metric name into one
  frame with a `run` key, captions naming every mark, CSV export, comparison pane.
- **The corpus** as scanned 2026-09-11: 113 recordings, 26 workloads, 4 campaigns, 64 channels,
  262144 pages per dump. So effective n is 26 workloads and a handful of families. This number
  governs the whole design: it is small.
- **Environment**: laptop M2 16 GB, python 3.10.12 (pyenv), has numpy 2.2.6, scipy,
  scikit-learn 1.7.2, PyWavelets, kymatio, and since 2026-09-17 **torch 2.14.0** (MPS available,
  4 threads; an LSTM over a `(64, 8, 32)` path batch ran). The server has numpy only (no pywt, no
  kymatio, no sklearn venv unless made).

## The three input shapes

Each Learn pipeline starts from one of these, taken from a run:

| shape | tensor per tile | where it comes from | what consumes it |
|---|---|---|---|
| **rows** | `(n_metrics,)` | `features.npz` as written today | tabular models, the AE bank, clustering, forests |
| **path** | `(W_t, n_blocks)` | after Block on the address axis, the per-block series stacked over the window: at each frame a vector over blocks, over the window a trajectory through block-space | sequence models (RNN, LSTM, GRU, TCN, transformers, ROCKET) |
| **image** | `(W_t, n_pages_active)` | `dense_page_matrix`, the page-by-time image the methodology describes | 2-D CNNs, conv AE/VAE, ViT/MAE, frozen image backbones |

"Path" is JK's word and he confirmed this reading (2026-09-17): a blocked tile *is* a path,
the sequence of block vectors over time. It can be drawn two ways: as a frames x blocks heatmap,
and as the most-active block per frame, a line walking through the address space. With the
default `wp = hp = 8192` pages over 262144 pages that is 32 blocks; W_t = 8 by default.

**Consequence for the runner (decided 2026-09-17):** the scheme gets a `Write tiles` output
(or a parameter on `Write`) that saves the tiles of the chosen shape beside `features.npz`, with
the same keys, under the same byte budget page-resolution tiles already obey. Today the tiles exist
only inside the executor's process. Measured sizes from the README: a dense page-resolution
tile set for the busiest workload is 171 MB at 700 pairs in `active` mode, 734 MB in `all`.

## The pipeline graph

Same machinery as the scheme canvas (palette, typed ports, inspector, validator with hard /
soft / note, examples), new tiers and new port types. The graph is the *data path only*; splits,
seeds, arms and the sweep live in the run panel, the way differ speed and max pairs live in the
Run tab today. The executor runs the graph once per (configuration, split, fold) and fits every
fitted node on that fold's training rows only, by construction.

Tiers (proposal):

1. **Input**: which run(s), which shape, which arm (APF / wAPF where the scheme carried both).
2. **Preprocess** (fitted on train only when it fits anything): scaler (standard, robust,
   min-max, quantile), log1p, PCA, kernel PCA, ICA, random projection, feature selection
   (variance, mutual information); for paths: differencing (the delta-as-derivative idea, so it
   deserves a module), per-block scaling, tapering, resampling; for images: log of counts, block
   pooling, `page_mode`. Two are **not** transforms: t-SNE and UMAP are for looking, never for a
   fitted feature that a test row passes through. Per-recording normalisation (z-score within
   the recording) is a scientific choice, not a convenience: it removes level information, which
   is exactly what the APF vs wAPF arms are about; offer it, warn softly, record it.
3. **Represent** (optional): an encoder whose output feeds a model: AE / VAE latent, contrastive
   embedding, a frozen pretrained encoder (see catalogue).
4. **Model**: see catalogue. Two wrappers apply to any reconstruction model: the **bank** (one
   model per family, argmin error = predicted family, B1 Phase 1) and the **one-class** novelty
   scorer (fit on the benign set, score everything).
5. **Score**: see the scores section. Emits a results frame.
6. **Write**: results, per-window scores, embeddings, confusion, sidecar. Without it the run is
   not valid, as with the scheme.

Port types (proposal): `rows`, `path`, `image` (the three shapes), `embedding` (rows again but
produced by a model), `labels` (family, workload, recording, implicit from keys), `scores`,
`results`. A model declares which shapes it accepts; the validator refuses a mismatch by name.

## Model catalogue

Organised by input shape, older to newer, with what each needs and what it is for. The
recency JK asked for is real, but the small n decides which tier is honest: with 26 workloads,
**pretrained or non-parametric methods with light heads are the honest tier**; deep models
trained from scratch are a ceiling probe that must be shown next to a null. Every model refuses
by name when its dependency is missing (the wavelet / scattering precedent), never silently.

**Rows (tabular)**

| model | year | needs | for |
|---|---|---|---|
| logistic regression, linear SVM, kNN | classic | sklearn | floors every other number is compared to |
| random forest, ExtraTrees, HistGradientBoosting | classic | sklearn | supervised foil (B1 Phase 2 has the forest) |
| k-means, GMM, hierarchical | classic | sklearn | do families emerge unsupervised (B1 Phase 2) |
| AE bank (MLP) | B1 | sklearn | per-family reconstruction, argmin (B1 Phase 1) |
| IsolationForest, LOF, one-class SVM | classic | sklearn | novelty scores |
| COPOD, ECOD | 2020, 2022 | pyod | parameter-free novelty scores |
| FT-Transformer | 2021 | torch | attention over features, a deep tabular probe |
| TabPFN v2 | 2025 | torch + weights | pretrained tabular foundation model, built for small n: the newest thing that actually fits this regime |

**Path (sequence, W_t x n_blocks)**

| model | year | needs | for |
|---|---|---|---|
| RNN, LSTM, GRU | classic | torch | the sequence models JK named |
| 1-D CNN, TCN | 2018 | torch | convolution over time |
| InceptionTime | 2020 | torch | the strong CNN ensemble for time-series classification |
| ROCKET, MiniRocket, MultiRocket, Hydra | 2020 to 2023 | numpy (aeon has them; MiniRocket is ~100 lines to hand-roll) | random convolution kernels + ridge: near state of the art on the UCR archive, minutes on CPU, no torch; the first newer model to add |
| Transformer encoder, PatchTST | 2023 | torch | attention over patches of the sequence |
| S4, Mamba | 2022, 2023 | torch (Mamba wants CUDA kernels; S4D runs on CPU) | state-space sequence models, the newest family; heavy |
| TS2Vec, TS-TCC | 2022, 2021 | torch | contrastive self-supervised embeddings, then a light head |
| Chronos, MOMENT, TimesFM | 2024 | torch + weight downloads (hundreds of MB) | pretrained time-series foundation models as frozen encoders, zero-shot embeddings + kNN / logreg: the honest way to use a big model at n = 26 |
| Deep SVDD, USAD, TranAD, Anomaly Transformer | 2018, 2020, 2022, 2022 | torch | novelty / anomaly scoring on sequences |

**Image (W_t x pages)**

| model | year | needs | for |
|---|---|---|---|
| small 2-D CNN (ResNet-style) | classic | torch | the CNN JK named |
| conv AE, conv VAE, beta-VAE, VQ-VAE | 2013 to 2017 | torch | reconstruction and latent structure of the page-time image |
| ConvNeXt | 2022 | torch (+ ImageNet weights, optional) | the modern CNN |
| ViT / DeiT, MAE | 2021, 2022 | torch (+ weights) | attention over patches of the image; MAE pretrains without labels, then a linear probe |
| frozen ImageNet backbone (ResNet-18, ConvNeXt-T) + head | classic | torch + weights | cheap, honest at small n: the image treated as grayscale |

**Any shape, generative**: AE, VAE, normalizing flows (RealNVP). **Diffusion is a locked slot**:
JK deferred it to its own paper (2026-09); the palette shows it as "deferred to its own study",
not as unbuilt.

**Recommended first tier** (proposal, so the view is useful before the heavy dependencies
land): everything sklearn above, MiniRocket in numpy, HistGradientBoosting, ECOD; then with torch
on the laptop: LSTM / GRU / TCN, conv AE / VAE on images, Deep SVDD; then the pretrained encoders
(one time-series foundation model, one ImageNet backbone, TabPFN); then PatchTST, S4, MAE, TS2Vec,
USAD / TranAD. Package names to verify before pinning: `torch`, `pyod`, `aeon`, `umap-learn`,
`tabpfn`, `chronos-forecasting`, `momentfm`, `timesfm`.

## Splits, and the discipline that does not bend

From B1, carried over whole: within-trace (ceiling, never the headline), leave-one-rep-out,
leave-one-workload-out (the headline), grouped so nothing straddles; scaler and every fitted
step on train only; k fixed in advance for clustering; one recorded seed per configuration;
label-shuffle null as the floor; both arms where the run carries both. **Add** (proposal):
leave-one-campaign-out, because the four campaigns were captured at different times and the E1
notes record a cell-order confound; a split that holds a campaign out tests whether the model
learned the campaign.

Sweeps: "execute every possibility" is the product of the chosen options per tier x splits x
seeds. It runs. It is also a multiple-comparisons hazard at n = 26, so the results carry the
number of configurations tried and the null next to every score, as a soft warning, not a
refusal. That is JK's own rule: a perfect score is a symptom.

## Scores, deeper than one number

Per configuration, per split, per fold, and aggregated:

- **Classification**: accuracy, balanced accuracy, macro F1, per-family and per-workload
  precision / recall / F1, Cohen's kappa, MCC, top-k, and the **confusion matrix**, which is the
  thesis's own object (the predicted-confusion question: does a kernel's shape get confused with
  a non-compute occupant).
- **Detection / novelty**: AUROC, AUPRC, recall at a fixed false-positive rate, the threshold
  curve, per-family score distributions, calibration (reliability, ECE, Brier).
- **Unsupervised**: ARI and NMI against family and against workload, silhouette,
  Davies-Bouldin, Calinski-Harabasz on the embedding, kNN purity in latent space, the
  cluster x family contingency table.
- **Reconstruction**: per-window error, the window x family error matrix (a B1 artifact to
  persist), per-family error distributions, and the arm difference APF vs wAPF on the same folds.
- **Honesty**: per-fold mean and std, bootstrap CI over recordings, paired difference between
  two configurations on the same folds with its CI, the label-shuffle null distribution and the
  observed score's percentile in it, learning curve (score vs number of training workloads),
  train vs test gap, wall time and memory, and the count of configurations tried.
- **Explanation** (which part of memory carries the identity, a thesis question): permutation
  importance (rows), tree importances, per-block importance (path), saliency / Grad-CAM /
  integrated gradients (image, sequence), attention maps (transformers), VAE latent traversals.
- **Over time**: the model's score along `t_index` per recording, so a run shows *where* in the
  recording it is confident; workloads carry phase markers, so phases can be aligned later.

## Visualizations: Explore, reused, plus the few learning needs

The results of Learn are a frame (rows keyed by configuration, split, fold, recording, family,
workload, t_index; columns are scores), so `results_view.py` and the five views apply as they
are: distribution of a score by configuration, score over `t_index`, a matrix of any score over
(configuration x split), a scatter of two scores, the table. The comparison pane compares
configurations the way it compares runs today.

Added views (proposal), each drawn by the same hand-rolled SVG machinery, numbers from the bridge:
confusion matrix (the matrix view with the true / predicted labelling and per-row recall);
ROC and PR curves (a line view with x = FPR, y = TPR, one line per configuration or family);
embedding scatter (the scatter view over a 2-D projection computed on the bridge: PCA always,
UMAP when installed, coloured by family, with the split's train / test marked); the null
distribution with the observed score marked; training curves (loss vs epoch, train and
validation, the over-time view); saliency and attention as a heatmap over the tile (the matrix
view over frames x blocks or frames x pages); the path itself (frames x blocks heatmap, and the
most-active block per frame as a line); VAE latent traversals as small multiples.

## Outputs and provenance

Each configuration is a run directory: `learn_results.json` (every score above), per-window
scores and predictions, embeddings, the error matrix, confusion, and a sidecar that names the
input run and the hash of *its* sidecar, the shape, the arm, the split and its folds
(recording ids per fold), the seed, every preprocessing and model parameter, library versions,
torch determinism flags, wall time. The artifacts B1 asked to persist for the next papers
(latents, the error matrix, cluster assignments and centroids) fall out of this for free.

## Where it runs

Laptop by default (torch is installed there; CPU or MPS on the M2 is enough for the small
models; the pretrained encoders run inference only). The server stays sklearn-only until JK says otherwise; a run that needs
torch on the server refuses by name. Weight downloads happen once, into a named cache, and are
reported with their size before the first download.

## Decisions

Decided by JK on 2026-09-17:

1. "different staring functions" means **scoring functions**. Scaling functions and seeds are
   present as options anyway.
2. The **path** reading above is right: the blocked tile as a sequence of block vectors.
3. **torch** is installed on the laptop (2.14.0, done the same day).
4. **Write tiles** in the runner is the way images and paths reach Learn.

Still open:

5. Which of the newer models are worth their dependency weight first. The recommended first
   tier above stands as the default until he says otherwise.
6. Per-recording normalisation offered as a preprocessing choice with a soft warning. Default:
   offered, warned, recorded.

## What not to do

Do not put learning modules on the scheme canvas: the scheme is per-recording extraction and its
sidecar is provenance for `features.npz`; learning crosses recordings and has folds. Do not sweep
silently. Do not let t-SNE or UMAP become a fitted feature. Do not build diffusion. Do not read,
grep, or name the sandbox kernels or their workloads anywhere in the view or its examples; say
"the sandbox family".
