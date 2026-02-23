import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PLV_KEYS = [
    "num_anomalous_drop",
    "num_very_weak_stability",
    "num_moderate_stability",
    "num_high_stability",
    "num_perfect_stability",
]


def load_results(path: str) -> dict:
    obj = np.load(path, allow_pickle=True)
    # saved as ndarray(shape=(), dtype=object) holding a dict
    if isinstance(obj, np.ndarray) and obj.shape == ():
        return obj.item()
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unexpected npy content type: {type(obj)}")


def run_label_from_path(path: str) -> str:
    stem = Path(path).stem
    # your files look like stability_features_A1, etc.
    return stem.replace("stability_features_", "")


def parse_wind_key(k: str) -> int:
    # "wind_128_256" -> 128
    m = re.match(r"wind_(\d+)_(\d+)$", k)
    if not m:
        return 10**9
    return int(m.group(1))


def plv_total_from_counts(d: dict) -> int:
    return int(sum(d.get(k, 0) for k in PLV_KEYS))


def summarize_full_run(res: dict) -> dict:
    out = {}

    # ---- PLV (counts + ratios) ----
    plv = res["full_run"]["plv"]
    total = plv_total_from_counts(plv)

    # ratios (kept for reference if you ever want them)
    out["plv_perfect_ratio"] = plv.get("num_perfect_stability", 0) / max(total, 1)
    out["plv_highplus_ratio"] = (
        (plv.get("num_high_stability", 0) + plv.get("num_perfect_stability", 0))
        / max(total, 1)
    )
    out["plv_anom_rate"] = plv.get("num_anomalous_drop", 0) / max(total, 1)

    # counts (preferred)
    out["plv_total"] = int(total)
    out["plv_perfect_count"] = int(plv.get("num_perfect_stability", 0))
    out["plv_high_count"] = int(plv.get("num_high_stability", 0))
    out["plv_moderate_count"] = int(plv.get("num_moderate_stability", 0))
    out["plv_very_weak_count"] = int(plv.get("num_very_weak_stability", 0))
    out["plv_anom_count"] = int(plv.get("num_anomalous_drop", 0))

    # ---- Cepstrum ----
    ceps_full = res["full_run"]["cepstrum"]
    out["cep_peak_median_full"] = float(ceps_full["cepstral_peak_idx_median"])

    # ---- MSC tail stats (important!) ----
    msc_full = res["full_run"]["msc"]
    snr = msc_full["msc_peak_snr_db_per_page"]  # array [N]
    out["msc_mean"] = float(np.mean(snr))
    out["msc_p95"] = float(np.quantile(snr, 0.95))
    out["msc_p99"] = float(np.quantile(snr, 0.99))
    out["msc_max"] = float(np.max(snr))

    return out


def sliding_series(res: dict):
    """
    Returns per-window series:
      - plv perfect_ratio
      - plv anomalous_rate
      - cep peak median
    """
    plv_w = res["sliding_windows_128_64"]["plv"]
    cep_w = res["sliding_windows_128_64"]["cepstrum"]

    wind_keys = sorted(plv_w.keys(), key=parse_wind_key)

    perfect = []
    anom = []
    cep_meds = []

    for k in wind_keys:
        plv = plv_w[k]
        total = plv_total_from_counts(plv)
        perfect.append(plv.get("num_perfect_stability", 0) / max(total, 1))
        anom.append(plv.get("num_anomalous_drop", 0) / max(total, 1))

        # cepstrum windows exist with same key names
        cm = float(cep_w[k]["cepstral_peak_idx_median"])
        cep_meds.append(cm)

    return (
        wind_keys,
        np.array(perfect, dtype=float),
        np.array(anom, dtype=float),
        np.array(cep_meds, dtype=float),
    )


def zscore(x: np.ndarray) -> np.ndarray:
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd == 0.0:
        return np.zeros_like(x, dtype=float)
    return (x - mu) / sd


def plot_run_order_signature(run_names, summaries, metric_keys):
    """Run-ordered signature plot.

    Fix for sparse/heavy-tailed counts:
      - If metric is count-like (endswith _count OR is plv_total),
        apply log1p(x) BEFORE z-scoring.

    Then z-score each metric across runs so lines are comparable.
    """
    plt.figure()
    x = np.arange(len(run_names))

    for mk in metric_keys:
        vals = np.array([summaries[r][mk] for r in run_names], dtype=float)

        is_count_like = mk.endswith("_count") or mk == "plv_total"
        if is_count_like:
            vals = np.log1p(vals)  # safe for zeros
            label = f"{mk} (log1p)"
        else:
            label = mk

        plt.plot(x, zscore(vals), marker="o", label=label)

    plt.xticks(x, run_names, rotation=0)
    plt.title("Run-ordered behavior signature (z-scored; log1p on counts)")
    plt.xlabel("Run (ordered like time)")
    plt.ylabel("z-score (relative across runs)")
    plt.legend()
    plt.tight_layout()


def plot_heatmap(run_names, series_dict, which="perfect"):
    """Heatmap: rows=runs, cols=sliding windows.

    Different runs can have different numbers of windows.
    We build a UNION of all window keys across runs and pad missing entries
    with NaN so the heatmap can still be stacked.

    which in {"perfect","anom","cep"}
    """
    # Union of window keys across runs
    all_winds = set()
    for r in run_names:
        winds, *_ = series_dict[r]
        all_winds.update(winds)

    union_winds = sorted(all_winds, key=parse_wind_key)
    wind_to_col = {k: i for i, k in enumerate(union_winds)}

    # [num_runs, num_union_windows]
    M = np.full((len(run_names), len(union_winds)), np.nan, dtype=float)

    for ri, r in enumerate(run_names):
        winds, perfect, anom, cep = series_dict[r]
        vec = {"perfect": perfect, "anom": anom, "cep": cep}[which]
        for k, v in zip(winds, vec):
            M[ri, wind_to_col[k]] = float(v)

    plt.figure()
    plt.imshow(M, aspect="auto", interpolation="nearest")
    plt.yticks(np.arange(len(run_names)), run_names)
    plt.xlabel("Sliding window index (union across runs)")
    plt.title(f"Heatmap across runs × windows: {which} (NaN = missing window)")
    plt.colorbar()
    plt.tight_layout()


def plot_violin_distributions(run_names, results_by_run):
    """
    Deep distribution plots (per-page arrays):
      - MSC peak SNR distribution per run
      - Cepstral peak index distribution per run (full run)
    """
    # MSC
    msc_data = [
        results_by_run[r]["full_run"]["msc"]["msc_peak_snr_db_per_page"]
        for r in run_names
    ]
    plt.figure()
    plt.violinplot(msc_data, showmeans=True, showextrema=True)
    plt.xticks(np.arange(1, len(run_names) + 1), run_names)
    plt.title("MSC per-page SNR distribution (full run)")
    plt.ylabel("msc_peak_snr_db_per_page")
    plt.tight_layout()

    # Cepstral peak indices
    cep_data = [
        results_by_run[r]["full_run"]["cepstrum"]["cepstral_peak_idx_per_page"]
        for r in run_names
    ]
    plt.figure()
    plt.violinplot(cep_data, showmeans=True, showextrema=True)
    plt.xticks(np.arange(1, len(run_names) + 1), run_names)
    plt.title("Cepstral peak index distribution (full run)")
    plt.ylabel("cepstral_peak_idx_per_page")
    plt.tight_layout()


def main(paths):
    print(f"Running: {__file__}")

    # load in the order you pass → that IS your “time axis”
    results_by_run = {}
    run_names = []

    for p in paths:
        name = run_label_from_path(p)
        run_names.append(name)
        results_by_run[name] = load_results(p)

    # summaries
    summaries = {r: summarize_full_run(results_by_run[r]) for r in run_names}

    # sliding series for heatmaps
    series_dict = {r: sliding_series(results_by_run[r]) for r in run_names}

    # Debug: window coverage per run
    print("\n=== Sliding window coverage ===")
    for r in run_names:
        winds, *_ = series_dict[r]
        if len(winds) == 0:
            print(f"{r:>3} | num_windows=0")
        else:
            print(f"{r:>3} | num_windows={len(winds)} first={winds[0]} last={winds[-1]}")

    # 1) Run-ordered signature — COUNTS for PLV + log1p on counts inside the plot
    plot_run_order_signature(
        run_names,
        summaries,
        metric_keys=[
            "plv_perfect_count",
            "plv_anom_count",
            "cep_peak_median_full",
            "msc_p99",
        ],
    )

    # 2) Heatmaps (deep temporal view)
    plot_heatmap(run_names, series_dict, which="perfect")
    plot_heatmap(run_names, series_dict, which="anom")
    plot_heatmap(run_names, series_dict, which="cep")

    # 3) Distribution plots (deep per-page view)
    plot_violin_distributions(run_names, results_by_run)

    # Optional: print a compact table of scalar summaries
    print("\n=== Run-level summaries ===")
    for r in run_names:
        s = summaries[r]
        print(
            f"{r:>3} | "
            f"PLV perfect={s['plv_perfect_count']} "
            f"anom={s['plv_anom_count']} total={s['plv_total']} | "
            f"CEP median(full)={s['cep_peak_median_full']:.1f} | "
            f"MSC p99={s['msc_p99']:.4f} max={s['msc_max']:.4f}"
        )

    plt.show()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python deep_behavior_plots.py A1.npy A2.npy A3.npy B1.npy B2.npy B3.npy"
        )
    main(sys.argv[1:])