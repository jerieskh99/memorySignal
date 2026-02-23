from __future__ import annotations

import re
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict, List, Tuple


PLV_KEYS = [
    "num_anomalous_drop",
    "num_very_weak_stability",
    "num_moderate_stability",
    "num_high_stability",
    "num_perfect_stability",
]


def load_results(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.lib.npyio.NpzFile):
        raise ValueError("Expected .npy pickled dict, got .npz")
    d = obj.item()
    if not isinstance(d, dict):
        raise ValueError("Loaded object is not a dict")
    return d


def _parse_window_key(k: str) -> Tuple[int, int]:
    """
    'wind_64_192' -> (64, 192)
    """
    m = re.match(r"wind_(\d+)_(\d+)$", k)
    if not m:
        raise ValueError(f"Bad window key: {k}")
    return int(m.group(1)), int(m.group(2))


def _sorted_sliding_windows(plv_windows: Dict[str, Any]) -> List[str]:
    keys = list(plv_windows.keys())
    return sorted(keys, key=lambda k: _parse_window_key(k)[0])


def extract_plv_counts(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a normalized structure:
    {
      "full_run": {key: int, ...},
      "win512":   [ {key:int...}, {key:int...} ],
      "sliding":  [ ("wind_0_128", {..}), ... ]
    }
    """
    out: Dict[str, Any] = {}

    out["full_run"] = results["full_run"]["plv"]

    out["win512"] = [w["plv"] for w in results.get("512_windows", [])]

    sliding = results.get("sliding_windows_128_64", {}).get("plv", {})
    if sliding:
        keys = _sorted_sliding_windows(sliding)
        out["sliding"] = [(k, sliding[k]) for k in keys]
    else:
        out["sliding"] = []

    return out


def extract_cepstrum_medians(results: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    out["full_run"] = results["full_run"]["cepstrum"]["cepstral_peak_idx_median"]

    out["win512"] = [
        w["cepstrum"]["cepstral_peak_idx_median"]
        for w in results.get("512_windows", [])
    ]

    sliding = results.get("sliding_windows_128_64", {}).get("cepstrum", {})
    if sliding:
        keys = _sorted_sliding_windows(sliding)
        out["sliding"] = [(k, sliding[k]["cepstral_peak_idx_median"]) for k in keys]
    else:
        out["sliding"] = []

    return out


def extract_msc_stats(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    We compute mean/std from per-page vector if present, otherwise rely on stored median.
    """
    def stats(msc_dict: Dict[str, Any]) -> Dict[str, float]:
        d: Dict[str, float] = {}
        if "msc_peak_snr_db_per_page" in msc_dict:
            x = msc_dict["msc_peak_snr_db_per_page"]
            d["mean"] = float(np.mean(x))
            d["std"] = float(np.std(x))
            d["min"] = float(np.min(x))
            d["max"] = float(np.max(x))
        d["median"] = float(msc_dict.get("msc_peak_snr_db_median", np.nan))
        return d

    out: Dict[str, Any] = {}
    out["full_run"] = stats(results["full_run"]["msc"])

    out["win512"] = [stats(w["msc"]) for w in results.get("512_windows", [])]

    # note: your sliding_windows_128_64 has a single 'msc' dict (not per-window)
    if "sliding_windows_128_64" in results and "msc" in results["sliding_windows_128_64"]:
        out["sliding_global"] = stats(results["sliding_windows_128_64"]["msc"])
    else:
        out["sliding_global"] = None

    return out


# ---------------- PLOTTING ----------------

def plot_plv_sliding(plv: Dict[str, Any], *, title: str = "PLV (sliding windows)") -> plt.Figure | None:
    sliding = plv["sliding"]
    if not sliding:
        print("No sliding windows PLV to plot.")
        return None

    labels = [k for k, _ in sliding]
    counts = np.array([[d.get(key, 0) for key in PLV_KEYS] for _, d in sliding], dtype=float)

    # Convert to proportions to compare behavior across runs
    totals = counts.sum(axis=1, keepdims=True)
    props = counts / np.maximum(totals, 1.0)

    x = np.arange(len(labels))

    fig = plt.figure()
    bottom = np.zeros(len(labels))
    for i, key in enumerate(PLV_KEYS):
        plt.bar(x, props[:, i], bottom=bottom, label=key)
        bottom += props[:, i]
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Proportion of pages")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    return fig


def plot_plv_anomalies(plv: Dict[str, Any], *, title: str = "PLV anomalous_drop count") -> plt.Figure | None:
    sliding = plv["sliding"]
    if not sliding:
        print("No sliding windows PLV to plot.")
        return None

    labels = [k for k, _ in sliding]
    y = [d.get("num_anomalous_drop", 0) for _, d in sliding]

    x = np.arange(len(labels))
    fig = plt.figure()
    plt.plot(x, y, marker="o")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("num_anomalous_drop")
    plt.title(title)
    plt.tight_layout()
    return fig


def plot_cepstrum_median(ceps: Dict[str, Any], *, title: str = "Cepstral peak median over windows") -> plt.Figure | None:
    sliding = ceps["sliding"]
    if not sliding:
        print("No sliding windows cepstrum to plot.")
        return None

    labels = [k for k, _ in sliding]
    y = [v for _, v in sliding]

    x = np.arange(len(labels))
    fig = plt.figure()
    plt.plot(x, y, marker="o")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("cepstral_peak_idx_median")
    plt.title(title)
    plt.tight_layout()
    return fig


def _sanitize_filename(s: str) -> str:
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s

def main(path: str) -> None:
    results = load_results(path)

    plv = extract_plv_counts(results)
    ceps = extract_cepstrum_medians(results)
    msc = extract_msc_stats(results)

    print("Full-run summary:")
    print("  PLV:", {k: plv["full_run"].get(k) for k in PLV_KEYS})
    print("  Cepstrum median:", ceps["full_run"])
    print("  MSC:", msc["full_run"])

    # Output directory for figures
    out_dir = Path("/Users/jeries/Desktop/projects/thesis/memorySignal/mem_sig/coherence_temp_spec_stability/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = Path(path).stem
    run_name = _sanitize_filename(run_name)

    fig1 = plot_plv_sliding(plv, title="PLV proportions (128/64 sliding windows)")
    fig2 = plot_plv_anomalies(plv, title="PLV anomalous_drop (128/64 sliding windows)")
    fig3 = plot_cepstrum_median(ceps, title="Cepstral peak median (128/64 sliding windows)")

    saved_any = False

    if fig1 is not None:
        save_path = out_dir / f"{run_name}__plv_props.png"
        fig1.savefig(str(save_path), dpi=200)
        print(f"Saved: {save_path}")
        saved_any = True

    if fig2 is not None:
        save_path = out_dir / f"{run_name}__plv_anomalies.png"
        fig2.savefig(str(save_path), dpi=200)
        print(f"Saved: {save_path}")
        saved_any = True

    if fig3 is not None:
        save_path = out_dir / f"{run_name}__cepstrum_median.png"
        fig3.savefig(str(save_path), dpi=200)
        print(f"Saved: {save_path}")
        saved_any = True

    if not saved_any:
        print("No figures were saved (likely because sliding windows are empty in this results file).")
    else:
        print(f"Saved figures to: {out_dir}")

    # plt.show()


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python plot_results.py path/to/results.npy")
    main(sys.argv[1])
