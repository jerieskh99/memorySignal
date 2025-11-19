

# Baseline XOR + 1-NN Usage Guide

This document explains how to use the Baseline XOR modules to:

1. Collect XORs during a memory-sampling session
2. Build a full XOR time series
3. Extract a single feature vector describing the entire run
4. Train and evaluate a 1-Nearest-Neighbor (1-NN) classifier

The pipeline is designed for **offline evaluation and sanity-checks**.
It does *not* require keeping any raw memory pages after XORs are computed.

---

## 📦 Modules Overview

### `BaselineXOR`

A minimal helper class that:

* Loads selected pages using `Selector`
* Computes the bytewise XOR between two snapshots
* Returns a matrix of shape `(P, B)`, where:

  * `P` = number of selected pages
  * `B` = bytes per page

This class does **not** extract features and does **not** do classification.

### `OneNNBaselineXOR`

A wrapper class that:

* Uses `BaselineXOR` to compute XORs
* Extracts features from a **full time series** of XORs (shape `(T, B_total)`)
* Trains a `Simple1NN` classifier using those feature vectors

This class contains all feature-extraction logic.

---

## 🧩 Data Flow Summary

```
(prev dump) ──┐
              ├── XOR → flattened XOR row → store
(curr dump) ──┘

Repeat for all time steps…

→ Stack XOR rows → full XOR time series (T × B_total)
→ Extract features (1 × 8)
→ Train or classify using 1-NN
```

You only compute XORs *during sampling*; everything else happens *after* the run is finished.

---

## 🛠️ Example: Collect XOR Time Series During Sampling

This is the code pattern you use inside your sampling script:

```python
from baseline_xor import OneNNBaselineXOR

model = OneNNBaselineXOR(pathBitmap, pathPrev="", pathCurr="", numPages=numPages)

xor_rows = []
prev_path = None

for curr_path in sequence_of_paths_from_powershell():
    if prev_path is not None:
        # Compute XOR between previous and current snapshots
        xor_step = model.calculateXor(prev_path, curr_path)   # shape: (P, B_page)

        # Flatten all pages → single time-step vector
        xor_flat = xor_step.reshape(-1)                       # shape: (B_total,)
        xor_rows.append(xor_flat[None, :])                    # store as 2D row

    prev_path = curr_path
```

After the sampling session ends:

```python
# Build full XOR time series
xor_series = np.vstack(xor_rows)            # shape: (T, B_total)

# Extract a single 8-D feature vector for this entire run
feat_vector = model.extract_features_from_series(xor_series)
```

`feat_vector` now represents this whole run (normal, malware, etc.).

---

## 🎯 Training the 1-NN Classifier

Repeat the above process for each labeled run:

```python
all_features = []
all_labels = []

for run_paths, label in training_runs:
    xor_series = collect_xor_series(run_paths)
    feat = model.extract_features_from_series(xor_series)
    all_features.append(feat)
    all_labels.append(label)

X_train = np.stack(all_features, axis=0)  # (N_runs, 8)
y_train = np.array(all_labels)
```

Then train:

```python
model.fit(X_train, y_train)
```

---

## 🔍 Predicting on New Runs

For a new sampling session:

```python
xor_series_test = collect_xor_series(test_run_paths)
feat_test = model.extract_features_from_series(xor_series_test)

y_pred = model.predict(feat_test[None, :])
print("Predicted class:", y_pred[0])
```

---

## 📁 Recommended Project Structure

```
your_project/
│
├── baseline_xor.py         # contains BaselineXOR + OneNNBaselineXOR
├── utils/
│   └── randPageSelector.py
├── simple1NN.py
├── run_sampling.py         # where you call calculateXor during sampling
└── README.md               # <— this file
```

---

## ✔️ What This Baseline Gives You

* Quick sanity-check classifier
* One 8-dimensional feature vector per full memory-evolution sequence
* Ability to compare runs (Normal vs. Malware vs. Other processes)
* Works even if raw dumps are deleted immediately after XOR
* Minimal storage requirements

