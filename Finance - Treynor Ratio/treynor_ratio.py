"""
Treynor Ratio
-------------
The Sharpe ratio divides excess return by total volatility, which mixes two
very different kinds of risk. Some of a portfolio's wiggle comes from the
market itself and some is its own private noise. If you hold the portfolio
inside a well diversified book, the private noise mostly cancels out and the
only risk you are truly paid to carry is the market exposure, measured by
beta. The Treynor ratio therefore divides excess return by beta instead of by
standard deviation, asking how much reward the portfolio earns per unit of
market risk it actually adds.

A higher Treynor ratio means more excess return for each unit of beta. Two
funds with the same Sharpe can have very different Treynor ratios when one of
them buries a lot of diversifiable noise inside its volatility.
"""

from __future__ import annotations

import numpy as np


def beta(asset_returns: list[float], market_returns: list[float]) -> float:
    """Return the asset's beta to the market.

    Beta is the covariance of the asset with the market divided by the
    variance of the market. A beta of one moves with the market, above one
    amplifies it, below one dampens it, and negative beta moves against it.
    """
    a = np.asarray(asset_returns, dtype=float)
    m = np.asarray(market_returns, dtype=float)
    if a.size != m.size:
        raise ValueError("asset and market series must be the same length")
    if a.size < 2:
        raise ValueError("need at least two observations")
    market_var = float(np.var(m, ddof=1))
    if market_var == 0:
        raise ValueError("market returns have zero variance")
    covariance = float(np.cov(a, m, ddof=1)[0, 1])
    return covariance / market_var


def treynor_ratio(
    asset_returns: list[float],
    market_returns: list[float],
    risk_free_rate: float = 0.0,
) -> float:
    """Return the Treynor ratio using per period returns.

    This is the mean excess return of the asset over the risk free rate,
    divided by its beta to the market. The risk free rate must be quoted per
    period, matching the return series. A near zero beta makes the ratio
    meaningless, so that case raises instead of returning a huge number.
    """
    b = beta(asset_returns, market_returns)
    if abs(b) < 1e-12:
        raise ValueError("beta is zero, the Treynor ratio is undefined")
    excess = float(np.mean(np.asarray(asset_returns, dtype=float))) - risk_free_rate
    return excess / b


def annualized_treynor_ratio(
    asset_returns: list[float],
    market_returns: list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Return the Treynor ratio scaled to an annual figure.

    Beta is dimensionless, so annualizing only scales the excess return in
    the numerator by the number of periods per year.
    """
    return treynor_ratio(asset_returns, market_returns, risk_free_rate) * periods_per_year


if __name__ == "__main__":
    print("Treynor Ratio")
    print("=" * 40)

    rng = np.random.default_rng(5)
    market = rng.normal(0.0004, 0.01, size=500)

    # A low beta fund with real skill and a high beta fund riding the market.
    skilled = 0.5 * market + rng.normal(0.0004, 0.004, size=500)
    leveraged = 1.5 * market + rng.normal(0.0, 0.004, size=500)

    for name, series in [("skilled low beta", skilled), ("leveraged", leveraged)]:
        b = beta(series.tolist(), market.tolist())
        tr = annualized_treynor_ratio(series.tolist(), market.tolist())
        print(f"\n{name}")
        print(f"  beta            {b:.2f}")
        print(f"  treynor (annual) {tr:.4f}")

    print("\nThe skilled fund earns more per unit of market risk carried.")
