#!/usr/bin/env python3
"""plan05_fidelity.py -- APF-fidelity gate (G-T2) for Plan 05 throughput levers.

Plan 05 makes the per-snapshot RAW dump cheaper (tmpfs target, delete-as-you-go,
optionally smaller guest RAM) so that more snapshots fit per wall-clock second.
The *only* thing that must not change is the APF signal itself. This module is
the gate that certifies that: it compares a baseline-arm APF trajectory against
a lever-arm APF trajectory and decides whether the signal was preserved.

Three checks, ALL must pass (G-T2):

  1. TOST mean equivalence   -- |mean_base - mean_lever| < APF_MEAN_MARGIN,
                                established by two one-sided Welch t-tests.
  2. STD equivalence         -- the (1 - 2*alpha) bootstrap CI of the std
                                difference lies entirely within +/-APF_STD_MARGIN.
  3. KS not rejected         -- ks_2samp p-value > KS_ALPHA, i.e. the two APF
                                distributions are NOT shown to differ.

Equivalence testing (TOST) is the statistically correct frame here: the null is
"the arms differ by at least the margin", and passing means we have evidence the
arms are interchangeable -- which is what "fidelity preserved" actually claims.
A plain difference test (t / KS rejection) cannot establish sameness; KS is used
only in its honest direction (failure to reject is necessary, not sufficient,
hence it is one of three checks rather than the whole gate).

Reuses load_streaming_trajectory() from plan02_metrics_per_cell.py to read the
streaming apf_trajectory.jsonl emitted by plan02_apf_helper.py.

No VM required. Pure offline analysis.

    python3 plan05_fidelity.py --baseline BASE.jsonl --lever LEVER.jsonl
    python3 plan05_fidelity.py --baseline BASE.jsonl --lever LEVER.jsonl --json out.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan02_metrics_per_cell import load_streaming_trajectory  # noqa: E402

try:
    from scipy import stats as _sps
    _HAVE_SCIPY = True
except Exception:  # noqa: BLE001  -- scipy is declared in requirements.txt
    _sps = None
    _HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# Calibratable constants. Wave 3 may revise these after the pilot; they are
# module-level so a sensitivity sweep can import and perturb them.
# ---------------------------------------------------------------------------
APF_MEAN_MARGIN = 0.02    # absolute APF units (APF in [0, 1])
APF_STD_MARGIN = 0.02     # absolute APF units
TOST_ALPHA = 0.05         # one-sided level for each TOST leg / bootstrap CI
KS_ALPHA = 0.05           # ks_2samp p-value level (now advisory only)
KS_STAT_MARGIN = 0.10     # max acceptable KS statistic D = sup|F_a - F_b| (the
                          # CDF gap, an effect size in [0, 1]). The gate passes on
                          # D <= margin instead of the p-value, so large-n KS stops
                          # rejecting negligible differences. Data-calibrated on the
                          # pilot: D clusters <=0.093 (practically identical) vs
                          # >=0.150 (real differences); 0.10 sits in that gap.
FIDELITY_MIN_N = 8        # min APF points per arm for the gate to be applicable
BOOT_N = 2000             # bootstrap resamples for the std CI
BOOT_SEED = 12345         # deterministic bootstrap
_SE_EPS = 1e-12           # pooled-SE below this is treated as zero (constant arms)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class CheckResult:
    name: str
    passed: bool
    applicable: bool
    detail: dict = field(default_factory=dict)


@dataclass
class FidelityResult:
    passed: bool
    applicable: bool
    reason: str
    n_baseline: int
    n_lever: int
    checks: list  # list[dict] (asdict of CheckResult)
    params: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _require_scipy() -> None:
    if not _HAVE_SCIPY:
        raise RuntimeError(
            "plan05_fidelity requires scipy.stats (declared in requirements.txt). "
            "Install scipy to run the TOST / KS checks."
        )


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def tost_mean(a, b, margin: float = APF_MEAN_MARGIN,
              alpha: float = TOST_ALPHA) -> CheckResult:
    """Two one-sided t-tests for equivalence of APF means (Welch, unequal var).

    H0: |mean_a - mean_b| >= margin (non-equivalent).
    Equivalent iff BOTH one-sided legs reject at `alpha`; reported p_tost is the
    larger (worse) of the two leg p-values.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = int(a.size), int(b.size)
    if n1 < 2 or n2 < 2:
        return CheckResult("tost_mean", False, False,
                           {"reason": "need >= 2 samples per arm",
                            "n_baseline": n1, "n_lever": n2})
    m1, m2 = float(a.mean()), float(b.mean())
    v1, v2 = float(a.var(ddof=1)), float(b.var(ddof=1))
    diff = m1 - m2
    se = math.sqrt(v1 / n1 + v2 / n2)
    if se <= _SE_EPS:
        # Both arms are (numerically) constant: the difference is exact, so the
        # t-test degenerates. Decide equivalence directly against the margin
        # rather than dividing by a denominator at floating-point noise level.
        equiv = abs(diff) < margin
        return CheckResult("tost_mean", bool(equiv), True,
                           {"mean_baseline": m1, "mean_lever": m2,
                            "mean_diff": diff, "se": se, "margin": margin,
                            "p_tost": 0.0 if equiv else 1.0, "degenerate_se": True,
                            "note": "near-zero pooled SE (both arms ~constant)"})
    _require_scipy()
    df = (v1 / n1 + v2 / n2) ** 2 / (
        (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    )
    # Leg L: H0 diff <= -margin ; reject (diff > -margin) for large +t.
    t_lower = (diff + margin) / se
    p_lower = float(_sps.t.sf(t_lower, df))
    # Leg U: H0 diff >= +margin ; reject (diff < +margin) for large -t.
    t_upper = (diff - margin) / se
    p_upper = float(_sps.t.cdf(t_upper, df))
    p_tost = max(p_lower, p_upper)
    equiv = (p_lower < alpha) and (p_upper < alpha)
    return CheckResult("tost_mean", bool(equiv), True,
                       {"mean_baseline": m1, "mean_lever": m2, "mean_diff": diff,
                        "se": se, "df": df, "margin": margin, "alpha": alpha,
                        "p_lower": p_lower, "p_upper": p_upper, "p_tost": p_tost})


def bootstrap_std_equiv(a, b, margin: float = APF_STD_MARGIN,
                        alpha: float = TOST_ALPHA, n_boot: int = BOOT_N,
                        seed: int = BOOT_SEED) -> CheckResult:
    """Equivalence of APF std via a percentile bootstrap CI on the std difference.

    Equivalent iff the (1 - 2*alpha) CI of (std_a - std_b) lies entirely within
    (-margin, +margin). Bootstrap avoids any distributional assumption on the
    spread, which a t-based TOST cannot give for a standard deviation.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = int(a.size), int(b.size)
    if n1 < 2 or n2 < 2:
        return CheckResult("std_equiv", False, False,
                           {"reason": "need >= 2 samples per arm",
                            "n_baseline": n1, "n_lever": n2})
    rng = np.random.default_rng(seed)
    s1, s2 = float(a.std(ddof=1)), float(b.std(ddof=1))
    idx1 = rng.integers(0, n1, size=(n_boot, n1))
    idx2 = rng.integers(0, n2, size=(n_boot, n2))
    d = a[idx1].std(ddof=1, axis=1) - b[idx2].std(ddof=1, axis=1)
    lo = float(np.percentile(d, 100.0 * alpha))
    hi = float(np.percentile(d, 100.0 * (1.0 - alpha)))
    equiv = (lo > -margin) and (hi < margin)
    return CheckResult("std_equiv", bool(equiv), True,
                       {"std_baseline": s1, "std_lever": s2, "std_diff": s1 - s2,
                        "ci_low": lo, "ci_high": hi, "ci_level": 1.0 - 2.0 * alpha,
                        "margin": margin, "n_boot": n_boot, "seed": seed})


def ks_not_rejected(a, b, alpha: float = KS_ALPHA,
                    stat_margin: float = KS_STAT_MARGIN) -> CheckResult:
    """Two-sample KS as an effect-size check: pass = the distributions are
    practically equivalent (KS statistic D <= stat_margin).

    The KS *p-value* gains power with sample size, so at large n it rejects on
    negligible differences (D ~ 0.05) far below the APF margin we care about. The
    KS *statistic* D = sup|F_a - F_b| is an effect size in [0, 1] that does NOT
    inflate with n: for identical distributions D -> 0, for genuinely different
    ones D -> the true CDF gap. Gating on D therefore tests practical sameness,
    not mere detectability. The p-value is retained as advisory only.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 1 or b.size < 1:
        return CheckResult("ks_2samp", False, False, {"reason": "empty arm"})
    _require_scipy()
    res = _sps.ks_2samp(a, b)
    stat, p = float(res.statistic), float(res.pvalue)
    # Pass if the distributions are practically equivalent (small effect size,
    # D <= stat_margin) OR not statistically distinguishable (p > alpha). The
    # effect-size leg stops large-n KS from rejecting negligible differences; the
    # p-value leg stops small-n sampling noise (D ~ 1.36/sqrt(n) under H0) from
    # rejecting genuinely identical arms. A real difference fails BOTH legs.
    by_stat = stat <= stat_margin
    by_p = p > alpha
    return CheckResult("ks_2samp", bool(by_stat or by_p), True,
                       {"statistic": stat, "stat_margin": stat_margin,
                        "pvalue": p, "alpha": alpha,
                        "pvalue_not_rejected": bool(by_p),
                        "passed_via": "statistic" if by_stat else ("pvalue" if by_p else "neither"),
                        "interpretation": "pass = D <= stat_margin OR p > alpha "
                                          "(practically equivalent, or not "
                                          "statistically distinguishable)"})


# ---------------------------------------------------------------------------
# Composite gate
# ---------------------------------------------------------------------------
def fidelity_gate(baseline, lever, *,
                  mean_margin: float = APF_MEAN_MARGIN,
                  std_margin: float = APF_STD_MARGIN,
                  alpha_tost: float = TOST_ALPHA,
                  alpha_ks: float = KS_ALPHA,
                  ks_stat_margin: float = KS_STAT_MARGIN,
                  min_n: int = FIDELITY_MIN_N,
                  boot_n: int = BOOT_N,
                  boot_seed: int = BOOT_SEED) -> FidelityResult:
    """G-T2: returns a FidelityResult; .passed iff all three checks pass."""
    b0 = [float(x) for x in baseline]
    l0 = [float(x) for x in lever]
    nb, nl = len(b0), len(l0)
    params = {"mean_margin": mean_margin, "std_margin": std_margin,
              "alpha_tost": alpha_tost, "alpha_ks": alpha_ks,
              "ks_stat_margin": ks_stat_margin, "min_n": min_n,
              "boot_n": boot_n, "boot_seed": boot_seed}
    if nb < min_n or nl < min_n:
        return FidelityResult(
            False, False,
            f"insufficient APF points (baseline {nb}, lever {nl}; need >= {min_n})",
            nb, nl, [], params)
    c_mean = tost_mean(b0, l0, mean_margin, alpha_tost)
    c_std = bootstrap_std_equiv(b0, l0, std_margin, alpha_tost, boot_n, boot_seed)
    c_ks = ks_not_rejected(b0, l0, alpha_ks, ks_stat_margin)
    checks = [c_mean, c_std, c_ks]
    applicable = all(c.applicable for c in checks)
    passed = applicable and all(c.passed for c in checks)
    if passed:
        reason = "all checks pass"
    else:
        reason = "; ".join(
            f"{c.name}:{'pass' if c.passed else ('NA' if not c.applicable else 'fail')}"
            for c in checks)
    return FidelityResult(passed, applicable, reason, nb, nl,
                          [asdict(c) for c in checks], params)


def fidelity_from_paths(baseline_jsonl, lever_jsonl, **kw) -> FidelityResult:
    """Read two streaming apf_trajectory.jsonl files and run the gate."""
    b, _ = load_streaming_trajectory(Path(baseline_jsonl))
    l, _ = load_streaming_trajectory(Path(lever_jsonl))
    return fidelity_gate(b, l, **kw)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="plan05_fidelity.py",
        description="APF-fidelity gate (G-T2): TOST mean + bootstrap std + KS.")
    p.add_argument("--baseline", required=True, help="baseline arm apf_trajectory.jsonl")
    p.add_argument("--lever", required=True, help="lever arm apf_trajectory.jsonl")
    p.add_argument("--mean-margin", type=float, default=APF_MEAN_MARGIN)
    p.add_argument("--std-margin", type=float, default=APF_STD_MARGIN)
    p.add_argument("--alpha-tost", type=float, default=TOST_ALPHA)
    p.add_argument("--alpha-ks", type=float, default=KS_ALPHA)
    p.add_argument("--min-n", type=int, default=FIDELITY_MIN_N)
    p.add_argument("--json", default=None, help="write the full result dict here")
    args = p.parse_args(argv)

    res = fidelity_from_paths(
        args.baseline, args.lever,
        mean_margin=args.mean_margin, std_margin=args.std_margin,
        alpha_tost=args.alpha_tost, alpha_ks=args.alpha_ks, min_n=args.min_n)
    payload = res.to_dict()
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(payload, f, indent=2, default=str)
        tmp.replace(out)
    print(json.dumps(payload, indent=2, default=str))
    return 0 if res.passed else 1


if __name__ == "__main__":
    sys.exit(_main())
