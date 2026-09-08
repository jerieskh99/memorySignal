#!/usr/bin/env python3
"""modules.py -- the module registry: what can be dropped on the canvas, with typed ports.

The single definition the console renders from and scheme.py validates
against. Ports carry a signal type; a pipe only connects where the type fits
(TYPES). Parameters carry their defaults; feature lists are read from the
code that implements them where it exists, and tagged `proposed` where it
does not:

  simple statistics  plan08_b1/b1_features.py FEAT (imported; stdlib module)
                     + duty_gt05 from plan05_campaign/extra_features.py (parsed)
  deep statistics    stat_pass_frac, cepstral_peak_idx, ceps_peak_snr_db from
                     plan03_metric_kernel.py METRIC_KEYS (parsed; that module
                     imports numpy so it is not imported here)
                     + cv_workingset, f1_phase, coverage_ratio from
                     behavior_families.py FULL (parsed; imports sklearn)
                     + n_boundaries (plan04_cusum.py) and tau, skew, kurtosis,
                     entropy, which are proposed (plan10 UX section 10.2)
  wavelet families   pywt.wavelist() if pywt is importable in the build env,
                     otherwise an empty list and a note saying so

Nothing here is a channel name; those come from channel_roster.py.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
QEMU_DIR = HERE.parent

# ---------------------------------------------------------------------------
# signal types carried by ports
# ---------------------------------------------------------------------------

TYPES = {
    "cells":     {"label": "cells",         "css": "--t-cells"},
    "field":     {"label": "field",         "css": "--t-field"},
    "complex":   {"label": "complex field", "css": "--t-complex"},
    "series":    {"label": "series",        "css": "--t-series"},
    "tiles":     {"label": "tiles",         "css": "--t-tiles"},
    "features":  {"label": "features",      "css": "--t-features"},
    "reference": {"label": "reference",     "css": "--t-reference"},
}

TIERS = [
    {"k": "source",    "idx": "01", "name": "Source",    "color": "#78838f"},
    {"k": "compose",   "idx": "02", "name": "Compose",   "color": "#0f8ba3"},
    {"k": "divide",    "idx": "03", "name": "Divide",    "color": "#2563eb"},
    {"k": "lens",      "idx": "04", "name": "Lens",      "color": "#d97706"},
    {"k": "reference", "idx": "05", "name": "Reference", "color": "#b8860b"},
    {"k": "output",    "idx": "06", "name": "Output",    "color": "#1c8a4e"},
]


# ---------------------------------------------------------------------------
# feature lists, from the code that implements them
# ---------------------------------------------------------------------------

def simple_features() -> list[dict]:
    if str(QEMU_DIR / "plan08_b1") not in sys.path:
        sys.path.insert(0, str(QEMU_DIR / "plan08_b1"))
    from b1_features import FEAT  # noqa: E402  (pure stdlib, per its docstring)
    feats = [{"name": f, "source": "plan08_b1/b1_features.py FEAT"} for f in FEAT]
    text = (QEMU_DIR / "plan05_campaign" / "extra_features.py").read_text()
    m = re.search(r"EXTRA\s*=\s*\[([^\]]*)\]", text)
    extra = re.findall(r'"(\w+)"', m.group(1)) if m else []
    if "duty_gt05" in extra:
        feats.append({"name": "duty_gt05", "source": "plan05_campaign/extra_features.py EXTRA", "flag": "duty_gt05"})
    return feats


def deep_features() -> list[dict]:
    out: list[dict] = []
    kernel = (QEMU_DIR / "plan03_metric_kernel.py").read_text()
    m = re.search(r"METRIC_KEYS\s*=\s*\(([^)]*)\)", kernel)
    keys = re.findall(r'"(\w+)"', m.group(1)) if m else []
    for k in ("stat_pass_frac", "cepstral_peak_idx", "ceps_peak_snr_db"):
        if k in keys:
            out.append({"name": k, "source": "plan03_metric_kernel.py METRIC_KEYS"})
    if (QEMU_DIR / "plan04_cusum.py").exists():
        out.append({"name": "n_boundaries", "source": "plan04_cusum.py detect_boundaries_cusum (count)"})
    for k in ("tau", "skew", "kurtosis", "entropy"):
        out.append({"name": k, "source": "proposed: plan10_analysis_console_UX.md section 10.2", "proposed": True})
    bf = (QEMU_DIR / "plan05_campaign" / "behavior_families.py").read_text()
    m = re.search(r"FULL\s*=\s*\[([^\]]*)\]", bf)
    full = re.findall(r'"(\w+)"', m.group(1)) if m else []
    for k, flag in (("cv_workingset", "cv_workingset"), ("f1_phase", "f1_phase"), ("coverage_ratio", "coverage_ratio")):
        if k in full:
            out.append({"name": k, "source": "plan05_campaign/behavior_families.py FULL", "flag": flag})
    return out


def wavelet_families() -> dict:
    """Discrete families only: wavedec cannot take a continuous one (morl, mexh, gaus*, ...)."""
    try:
        import pywt  # type: ignore
        disc = list(pywt.wavelist(kind="discrete"))
        return {"families": disc, "source": f"pywt {pywt.__version__} wavelist(kind='discrete')",
                "excluded_continuous": sorted(set(pywt.wavelist()) - set(disc)),
                # the page has no pywt: it derives the level ceiling from the filter length
                "filter_len": {f: int(pywt.Wavelet(f).dec_len) for f in disc}}
    except ImportError:
        return {"families": [], "source": "pywt not importable in the build environment; family is free text",
                "excluded_continuous": [], "filter_len": {}}


# ---------------------------------------------------------------------------
# the modules
# ---------------------------------------------------------------------------

def scattering_limits() -> dict:
    """Max J per (window, Q), measured from kymatio so the page can enforce it without kymatio."""
    try:
        import kymatio  # type: ignore
    except ImportError:
        return {"available": False, "source": "kymatio not importable in the build environment", "max_J": {}}
    import sys as _sys
    _sys.path.insert(0, str(HERE / "runner"))
    from plan10_analysis.runner.stages import scattering_max_J  # noqa: E402
    grid = {str(w): {str(q): scattering_max_J(w, q) for q in (1, 2, 4, 8, 16)}
            for w in (8, 16, 32, 64, 128, 256, 512, 1024)}
    return {"available": True, "source": f"kymatio {kymatio.__version__} numpy frontend, measured", "max_J": grid}


def build_modules() -> dict:
    simple = simple_features()
    deep = deep_features()
    wav = wavelet_families()
    scat = scattering_limits()
    simple_default = [f["name"] for f in simple if "flag" not in f]
    deep_default = [f["name"] for f in deep if "flag" not in f]
    wav_opts = [[w, w] for w in wav["families"]]

    def mod(id, tier, name, ico, desc, inputs=(), outputs=(), params=(), flags=(), spectral=False, impl=None):
        return {"id": id, "tier": tier, "name": name, "ico": ico, "desc": desc,
                "inputs": list(inputs), "outputs": list(outputs), "params": list(params),
                "flags": list(flags), "spectral": spectral, "impl": impl}

    def inp(k, types, req=True, lab=None, multi=False):
        return {"k": k, "t": list(types), "req": req, "lab": lab or k, "multi": multi}

    def outp(k, t, lab=None):
        return {"k": k, "t": t, "lab": lab or TYPES[t]["label"]}

    def num(k, lab, default, step=None):
        return {"k": k, "kind": "number", "lab": lab, "default": default, "step": step}

    def sel(k, lab, opts, default):
        return {"k": k, "kind": "select", "lab": lab, "opts": opts, "default": default}

    def multi(k, lab, opts, default):
        return {"k": k, "kind": "multi", "lab": lab, "opts": opts, "default": default}

    M = [
        mod("cells", "source", "Cells", "C", "which recordings feed the scheme",
            outputs=[outp("cells", "cells")],
            params=[{"k": "sel", "kind": "cells", "lab": "recordings", "default": []},
                    num("min_pairs", "minimum pairs per recording", 50),
                    num("max_pairs", "max pairs per recording (0 = all; bounds a run, recorded in the sidecar)", 0)]),
        mod("channels", "source", "Channels", "ch",
            "pull the chosen metric columns out of each recording's substrate CSV, one streaming pass",
            inputs=[inp("cells", ["cells"])], outputs=[outp("field", "field")],
            params=[{"k": "chans", "kind": "channels", "lab": "channels", "default": []}]),
        mod("single", "compose", "Single", "1", "one channel becomes a scalar per page",
            inputs=[inp("in", ["field"])], outputs=[outp("out", "field")]),
        mod("vectorize", "compose", "Vectorize", "[]",
            "N channels become a real vector per page. No cross-channel math: the control arm the thesis never ran",
            inputs=[inp("in", ["field"])], outputs=[outp("out", "field", "field[N]")]),
        mod("complex", "compose", "Complex", "z",
            "one magnitude times one direction: magnitude and angle in one object",
            inputs=[inp("mag", ["field"], lab="magnitude"), inp("dir", ["field"], lab="direction")],
            outputs=[outp("out", "complex")],
            params=[sel("phase", "phase convention", [["", "choose"], ["2pi", "2 pi x distance"], ["pi", "pi x distance"], ["arccos", "arccos(similarity)"]], "")]),
        mod("collapse", "divide", "Collapse pages", "S",
            "average the page axis away: what every result to date does (APF is this)",
            inputs=[inp("in", ["field", "complex"])], outputs=[outp("out", "series")],
            params=[sel("reduce", "reduction", [["mean", "mean of the channel values over pages"], ["changed_fraction", "fraction of pages changed: K/N, which is APF"]], "mean"),
                    sel("unchanged", "unchanged pages (mean only)", [["zero", "count as zero (matches K/N)"], ["excluded", "excluded from statistics"]], "zero")]),
        mod("block", "divide", "Block pages", "#", "cut the address axis into blocks of whole pages",
            inputs=[inp("in", ["field", "complex"])], outputs=[outp("out", "field", "field, blocked")],
            params=[num("wp", "block W_p (pages)", 8192, 256), num("hp", "block hop H_p", 8192, 256)],
            flags=["gate2"]),
        mod("window", "divide", "Window time", "W", "slide a window over snapshots: W frames, hop H",
            inputs=[inp("in", ["series", "field", "complex"])], outputs=[outp("out", "tiles")],
            params=[num("w", "W_t (frames)", 8), num("h", "H_t (hop)", 4),
                    sel("edge", "edge handling (changes tile count)", [["drop", "drop: whole windows only"], ["zero", "zero-pad"], ["reflect", "reflect"], ["replicate", "replicate"]], "drop"),
                    sel("taper", "taper (changes spectral leakage)", [["rectangular", "rectangular: none"], ["hann", "hann"], ["hamming", "hamming"], ["blackman", "blackman"]], "rectangular")]),
        mod("stats", "lens", "Simple statistics", "x", "the scale-equivariant shape features already implemented",
            inputs=[inp("in", ["tiles", "series"])], outputs=[outp("out", "features")],
            params=[multi("feats", "features", simple, simple_default)], impl="plan08_b1/b1_features.py"),
        mod("deep", "lens", "Deep statistics", "t",
            "stationarity, boundaries, moments, entropy, and tau: the decorrelation time the per-channel window question rests on",
            inputs=[inp("in", ["tiles", "series"])], outputs=[outp("out", "features")],
            params=[multi("feats", "features", deep, deep_default)], impl="plan03_metric_kernel.py, plan04_cusum.py, proposed"),
        mod("fft", "lens", "FFT", "~", "spectrum, peak, band energies", spectral=True,
            inputs=[inp("in", ["tiles"])], outputs=[outp("out", "features")],
            params=[sel("out", "emit", [["bands", "band energies"], ["peak", "peak frequency"], ["spectrum", "full spectrum"]], "bands"),
                    sel("detrend", "detrend (without it a taper's DC leakage owns the peak)", [["mean", "subtract each tile's mean"], ["none", "raw tile"]], "mean")]),
        mod("cepstrum", "lens", "Cepstrum", "c", "peak quefrency and ceps_peak_snr_db, on the raw tile (a taper's envelope would own the quefrency)", spectral=True,
            inputs=[inp("in", ["tiles"])], outputs=[outp("out", "features")], flags=["ceps_snr"],
            impl="coherence_temp_spec_stability/cepstrum_stability.py CepstrumStability"),
        mod("wavelet", "lens", "Wavelet", "w", "pywt DWT / CWT", spectral=True,
            inputs=[inp("in", ["tiles"])], outputs=[outp("out", "features")],
            params=[sel("fam", "family (" + wav["source"] + ")", wav_opts, "db4" if any(o[0] == "db4" for o in wav_opts) else (wav_opts[0][0] if wav_opts else "")),
                    num("levels", "levels (the family and window set the ceiling)", 2),
                    sel("mode", "extension mode", [["periodization", "periodization: energy-preserving"], ["symmetric", "symmetric: pywt default, pads"]], "periodization")],
            flags=[] if wav["families"] else ["needs_env"], impl="pywt.wavedec"),
        mod("scattering", "lens", "Scattering", "S",
            "time-averaged 1D scattering, one feature per path; translation invariant by construction", spectral=True,
            inputs=[inp("in", ["tiles"])], outputs=[outp("out", "features")],
            params=[num("J", "J (the window and Q set the ceiling)", 2), num("Q", "Q (wavelets per octave)", 4)],
            flags=[] if scat["available"] else ["needs_env"], impl="kymatio.numpy.Scattering1D"),
        mod("msc", "lens", "MSC", "g", "magnitude-squared coherence between channels; re-windows internally (Welch)", spectral=True,
            inputs=[inp("in", ["tiles"])], outputs=[outp("out", "features")],
            params=[num("iw", "internal window", 128), num("ih", "internal step", 64)],
            impl="coherence_temp_spec_stability/magnitude_squared_coherence.py"),
        mod("plv", "lens", "PLV", "p", "phase-locking value against a fitted baseline; reads the phase of a complex signal",
            inputs=[inp("in", ["tiles"]), inp("ref", ["reference"], lab="baseline")], outputs=[outp("out", "features")],
            params=[num("drop", "drop threshold", 0.2), num("normal", "normal threshold", 0.7)],
            impl="coherence_temp_spec_stability/plv_calcolator.py"),
        mod("cusum", "lens", "CUSUM", "d", "change points on the series; boundary count",
            inputs=[inp("in", ["series", "tiles"])], outputs=[outp("out", "features")],
            params=[num("k", "k", 2.0), num("h", "h", 4.0)], flags=["cusum_band"], impl="plan04_cusum.py"),
        mod("baseline", "reference", "Baseline", "b", "fit a reference on a clean run (PLV) or a benign envelope (p5 to p95)",
            inputs=[inp("in", ["tiles", "series", "complex", "field"], lab="clean run")], outputs=[outp("out", "reference")],
            params=[sel("mode", "mode", [["cell", "single clean recording"], ["benign", "benign set: p5 to p95 envelope"]], "cell"),
                    {"k": "recording", "kind": "select", "lab": "clean recording (empty = first selected)", "opts": [], "default": ""}],
            impl="coherence_temp_spec_stability/plv_calcolator.py, plan05_campaign/normal_profile.py"),
        mod("concat", "output", "Concat features", "+", "join feature blocks side by side, one row per tile",
            inputs=[inp("in", ["features"], multi=True)], outputs=[outp("out", "features")]),
        mod("write", "output", "Write", ">", "npz plus csv twin, and the sidecar that records every choice; without it the run is not valid",
            inputs=[inp("in", ["features"], multi=True)],
            params=[sel("fmt", "format", [["npz+csv", "npz + csv twin"], ["npz", "npz only"], ["parquet+csv", "parquet + csv twin"]], "npz+csv")]),
    ]
    unbuilt = [{"tier": "compose", "name": "Other combiners, gates",
                "desc": "ratios, products, per-channel gating: designed, unbuilt (Entry 13 figure)"}]
    return {"schema": "plan10.modules.v1", "types": TYPES, "tiers": TIERS, "modules": M, "unbuilt": unbuilt,
            "feature_sources": {"simple": simple, "deep": deep, "wavelet": wav, "scattering": scat}}


if __name__ == "__main__":
    import json
    print(json.dumps(build_modules(), indent=1))
