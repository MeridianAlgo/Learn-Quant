"""
Extreme Value Theory (EVT) for Tail Risk
----------------------------------------
Normal-distribution risk models systematically underestimate big losses because
real returns have *fat tails*. EVT models the tail directly instead of the whole
distribution.

This module implements the **Peaks-Over-Threshold (POT)** approach: pick a high
threshold ``u``, model the *excess* losses above it with a Generalised Pareto
Distribution (GPD), and read VaR / Expected Shortfall off the fitted tail. It
also includes the **Hill estimator** for the tail index of heavy-tailed data.

Everything is implemented with NumPy only — the GPD is fit by the method of
moments, which is closed-form and robust for ``xi < 0.5`` (the regime that
covers essentially all financial return tails).
"""

from __future__ import annotations

from typing import Union

import numpy as np

ArrayLike = Union[list, np.ndarray]


def fit_gpd_mom(excesses: ArrayLike) -> dict:
    """Fit a Generalised Pareto Distribution to threshold excesses.

    Uses the method of moments (Hosking & Wallis). For excesses ``y`` with
    sample mean ``m`` and variance ``s2``:

        xi   = 0.5 * (1 - m^2 / s2)
        beta = 0.5 * m * (m^2 / s2 + 1)

    Args:
        excesses: Positive excess values ``y_i = loss_i - u`` for losses above
            the threshold ``u``.

    Returns:
        dict with shape ``xi`` and scale ``beta``.
    """
    y = np.asarray(excesses, dtype=float)
    y = y[y > 0]
    if y.size < 5:
        raise ValueError("need at least 5 positive excesses to fit a GPD")
    m = float(y.mean())
    s2 = float(y.var(ddof=1))
    if s2 <= 0:
        raise ValueError("excess variance must be positive")
    ratio = m * m / s2
    xi = 0.5 * (1.0 - ratio)
    beta = 0.5 * m * (ratio + 1.0)
    return {"xi": float(xi), "beta": float(beta), "n_excess": int(y.size), "threshold_mean_excess": m}


def pot_var_es(
    losses: ArrayLike,
    confidence: float = 0.99,
    threshold_quantile: float = 0.90,
) -> dict:
    """EVT estimate of Value-at-Risk and Expected Shortfall via POT.

    Args:
        losses: Loss series (positive numbers are losses). For returns ``r``,
            pass ``-r`` so losses are positive.
        confidence: VaR/ES confidence level, e.g. 0.99 for 99%.
        threshold_quantile: Quantile used to set the POT threshold ``u``
            (0.90 keeps the worst 10% of losses for the tail fit).

    Returns:
        dict with ``var``, ``es``, the fitted GPD parameters and the threshold.

    The closed-form POT formulas (McNeil, Frey & Embrechts) are::

        VaR_q = u + (beta/xi) * [ (n/Nu * (1 - q))^(-xi) - 1 ]
        ES_q  = VaR_q / (1 - xi) + (beta - xi*u) / (1 - xi)
    """
    x = np.asarray(losses, dtype=float)
    n = x.size
    if n < 50:
        raise ValueError("POT needs a reasonable sample; use >= 50 observations")

    u = float(np.quantile(x, threshold_quantile))
    excess = x[x > u] - u
    fit = fit_gpd_mom(excess)
    xi, beta = fit["xi"], fit["beta"]
    nu = fit["n_excess"]

    q = confidence
    tail_factor = (n / nu) * (1.0 - q)
    if abs(xi) < 1e-8:
        var = u + beta * (-np.log(tail_factor))
    else:
        var = u + (beta / xi) * (tail_factor ** (-xi) - 1.0)

    if xi < 1.0:
        es = var / (1.0 - xi) + (beta - xi * u) / (1.0 - xi)
    else:
        es = float("inf")  # mean does not exist for xi >= 1

    return {
        "var": float(var),
        "es": float(es),
        "threshold": u,
        "xi": xi,
        "beta": beta,
        "n_excess": nu,
        "confidence": q,
    }


def hill_estimator(data: ArrayLike, k: int) -> float:
    """Hill estimator of the tail index for heavy-tailed (xi > 0) data.

    Uses the top ``k`` order statistics of the (positive) data::

        xi_hat = (1/k) * sum_{i=1..k} [ ln X_(n-i+1) - ln X_(n-k) ]

    A larger ``xi_hat`` means a heavier tail (e.g. ~0.3-0.5 for equity losses).
    """
    x = np.asarray(data, dtype=float)
    x = np.sort(x[x > 0])
    n = x.size
    if k < 2 or k >= n:
        raise ValueError("require 2 <= k < n positive observations")
    top = x[n - k :]
    threshold = x[n - k - 1] if n - k - 1 >= 0 else x[0]
    return float(np.mean(np.log(top) - np.log(threshold)))


def mean_excess(losses: ArrayLike, thresholds: ArrayLike) -> np.ndarray:
    """Mean excess function e(u) = E[X - u | X > u] across thresholds.

    A roughly linear, upward-sloping mean-excess plot is the classic
    diagnostic that the GPD tail (with ``xi > 0``) is appropriate.
    """
    x = np.asarray(losses, dtype=float)
    out = []
    for u in np.asarray(thresholds, dtype=float):
        tail = x[x > u]
        out.append(float((tail - u).mean()) if tail.size else np.nan)
    return np.array(out)


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    # Student-t returns: fat tails, the whole point of EVT.
    dof = 4
    returns = 0.0005 + 0.012 * rng.standard_t(dof, size=4000)
    losses = -returns  # positive = loss

    print("Extreme Value Theory — Tail Risk")
    print("=" * 44)
    for c in (0.95, 0.99, 0.995):
        evt = pot_var_es(losses, confidence=c, threshold_quantile=0.90)
        # Compare to the naive Gaussian VaR.
        gauss = losses.mean() + losses.std(ddof=1) * {0.95: 1.645, 0.99: 2.326, 0.995: 2.576}[c]
        print(
            f"{int(c*100)}%  EVT VaR = {evt['var']:.4f}  ES = {evt['es']:.4f}"
            f"   |  Gaussian VaR = {gauss:.4f}"
        )

    fit = pot_var_es(losses, confidence=0.99)
    print(f"\nFitted GPD tail: xi = {fit['xi']:.3f}, beta = {fit['beta']:.4f} "
          f"({fit['n_excess']} excesses over u = {fit['threshold']:.4f})")
    print(f"Hill tail index (k=200): {hill_estimator(losses, 200):.3f}")
    print("\nNote how EVT VaR/ES exceed the Gaussian estimate in the deep tail —")
    print("that gap is the risk a normal model hides.")
