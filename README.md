# Correctness Validation for Volatile Memory Signal Model

This repository contains small, focused utilities used to verify the correctness of the page-wise differential encoding used in the volatile memory signal model.

The primary goal of these checks is to ensure that the **magnitude component** (Hamming-based bit flip count) is **consistent, symmetric, and complete** across incremental memory snapshots.

## Files

- **flip_accounting_check.py** — Verifies that bit-flip accounting is correct:
  - Checks that forward and backward comparisons are symmetric.
  - Confirms that the sum of changes matches expected flip counts.

## Requirements

Create and activate the virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate 
```

Then install dependencies (after generating requirements.txt):

```bash
pip install -r requirements.txt
```

## Usage

```bash
python flip_accounting_check.py --prev <path_to_previous_snapshot> --curr <path_to_current_snapshot>
```

Snapshots must be page-aligned and of the same size.

## Purpose

This repository is used **only for correctness validation** — not full analysis, feature extraction, or modeling. It allows verifying that the differential signal representation behaves exactly as intended before using it in larger pipelines.