import numpy as np
from pathlib import Path


def load_bitmap(path: str, num_pages: int)->np.ndarray:
    """
    Load a packed bitmap emitted by Rust (1 bit per page).
    Returns a NumPy array of shape (num_pages,), dtype=bool.
    """
    data = np.fromfile(path, dtype=np.uint8)
    bits = np.unpackbits(data)
    return bits[:num_pages].astype(bool)


def active_page_indices(bitmap: np.ndarray) -> np.ndarray:
    """Return indices of active (changed) pages."""
    return np.flatnonzero(bitmap)


def inactive_page_indices(bitmap: np.ndarray) -> np.ndarray:
    """Return indices of inactive (changed) pages."""
    return np.flatnonzero(~bitmap)


def active_ratio(bitmap: np.ndarray) -> float:
    """Return fraction of active pages."""
    return float((bitmap.sum()) / len(bitmap))

