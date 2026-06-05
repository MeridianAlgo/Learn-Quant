"""
Value at Risk (VaR) & Conditional VaR
-------------------------------------
VaR answers: "over the next period, how much could I lose, and with what
probability?" A 95% 1-day VaR of $1,000 means: on 95% of days losses should be
smaller than $1,000 — and on the worst 5% of days they are larger.

This module implements the three textbook estimators plus their tail companion
and a backtest:

* **Parametric (variance–covariance)** — assumes normal returns.
* **Historical** — uses the empirical return distribution, no shape assumption.
* **Monte Carlo** — simulates returns from a fitted model.
* **Conditional VaR / Expected Shortfall** — the *average* loss beyond VaR.
* **Kupiec POF test** — checks whether realised exceptions match the VaR level.

All functions return VaR as a **positive** number representing a loss.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import scipy.stats as stats

ArrayLike = Union[list, np.ndarray]


def value_at_risk(returns: ArrayLike, confidence_level: float = 0.95) -> float:
    """Parametric (normal) VaR — kept as the original public entry point."""
    return parametric_var(returns, confidence_level)


def parametric_var(returns: ArrayLike, confidence_level: float = 0.95) -> float:
    """Variance–covariance VaR assuming normally distributed returns.

    VaR = -(mu + sigma * z_alpha), where z_alpha = Phi^{-1}(1 - confidence).
    """
    r = np.asarray(returns, dtype=float)
    mu, sigma = r.mean(), r.std(ddof=1)
    alpha = 1.0 - confidence_level
    return float(-(mu + sigma * stats.norm.ppf(alpha)))


def historical_var(returns: ArrayLike, confidence_level: float = 0.95) -> float:
    """Historical-simulation VaR: the empirical quantile of the loss tail.

    Makes no distributional assumption — it simply reads the loss off the
    sorted return history, so it captures skew and fat tails the parametric
    method misses.
    """
    r = np.asarray(returns, dtype=float)
    alpha = 1.0 - confidence_level
    return float(-np.quantile(r, alpha))


def monte_carlo_var(
    returns: ArrayLike,
    confidence_level: float = 0.95,
    n_sims: int = 100_000,
    seed: int | None = None,
) -> float:
    """Monte Carlo VaR by simulating returns from a fitted normal model."""
    r = np.asarray(returns, dtype=float)
    rng = np.random.default_rng(seed)
    sims = rng.normal(r.mean(), r.std(ddof=1), n_sims)
    alpha = 1.0 - confidence_level
    return float(-np.quantile(sims, alpha))


def conditional_var(returns: ArrayLike, confidence_level: float = 0.95) -> float:
    """Conditional VaR (Expected Shortfall): mean loss *given* VaR is breached.

    A coherent risk measure — unlike VaR it is sub-additive, so diversification
    never increases it.
    """
    r = np.asarray(returns, dtype=float)
    alpha = 1.0 - confidence_level
    threshold = np.quantile(r, alpha)
    tail = r[r <= threshold]
    if tail.size == 0:
        return float(-threshold)
    return float(-tail.mean())


def kupiec_pof_test(returns: ArrayLike, var_level: float, confidence_level: float = 0.95) -> dict:
    """Kupiec proportion-of-failures backtest of a VaR estimate.

    Counts how often realised losses exceeded ``var_level`` and runs a
    likelihood-ratio test against the expected exception rate ``1 - confidence``.

    Returns the exception count, the LR statistic, its p-value and a
    ``reject`` flag (at 5%). A *good* VaR model is one we **fail to reject**.
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    exceptions = int(np.sum(r < -var_level))
    p = 1.0 - confidence_level
    pi = exceptions / n if n else 0.0

    # Likelihood ratio for unconditional coverage.
    if exceptions == 0:
        lr = -2.0 * n * np.log(1.0 - p)
    elif exceptions == n:
        lr = -2.0 * n * np.log(p)
    else:
        lr = -2.0 * (
            (n - exceptions) * np.log(1.0 - p)
            + exceptions * np.log(p)
            - (n - exceptions) * np.log(1.0 - pi)
            - exceptions * np.log(pi)
        )
    p_value = 1.0 - stats.chi2.cdf(lr, df=1)
    return {
        "exceptions": exceptions,
        "expected": p * n,
        "exception_rate": pi,
        "lr_statistic": float(lr),
        "p_value": float(p_value),
        "reject": bool(p_value < 0.05),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    daily_returns = rng.normal(0.0005, 0.02, 1000)

    print("Value at Risk — three methods (95% & 99%)")
    print("=" * 48)
    for c in (0.95, 0.99):
        print(
            f"{int(c * 100)}%  "
            f"parametric={parametric_var(daily_returns, c):.4f}  "
            f"historical={historical_var(daily_returns, c):.4f}  "
            f"monte_carlo={monte_carlo_var(daily_returns, c, seed=1):.4f}  "
            f"CVaR={conditional_var(daily_returns, c):.4f}"
        )

    var95 = parametric_var(daily_returns, 0.95)
    test = kupiec_pof_test(daily_returns, var95, 0.95)
    print("\nKupiec backtest of the 95% parametric VaR:")
    print(f"  exceptions {test['exceptions']} vs expected {test['expected']:.1f}")
    print(
        f"  LR = {test['lr_statistic']:.3f}, p = {test['p_value']:.3f} "
        f"=> {'REJECT' if test['reject'] else 'OK (model adequate)'}"
    )
