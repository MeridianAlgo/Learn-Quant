"""
Jensen's Alpha
--------------
Beta tells you how much of a portfolio's return came from simply riding the
market. Jensen's alpha tells you what is left over once that free ride is
subtracted. Take the return the CAPM says a portfolio should have earned
given its beta, subtract it from what the portfolio actually earned, and the
remainder is the manager's contribution. Positive alpha means skill or luck,
negative alpha means the fund would have done better as a cheap index bet
with the same beta.

Alpha is small and noisy, so a single positive number proves very little. The
t statistic here divides the alpha by the standard error of the regression
residuals, which is the honest way to ask whether the edge is real or whether
it is the kind of number a coin flipping manager produces every few years.
"""

from __future__ import annotations

import numpy as np


def beta(asset_returns: list[float], market_returns: list[float]) -> float:
    """Return the asset's beta to the market.

    Beta is the covariance of the asset with the market divided by the
    variance of the market, the slope of the line fitted through the scatter
    of asset returns against market returns.
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
    return float(np.cov(a, m, ddof=1)[0, 1]) / market_var


def expected_return(
    asset_returns: list[float],
    market_returns: list[float],
    risk_free_rate: float = 0.0,
) -> float:
    """Return the per period return the CAPM predicts for this asset.

    The prediction is the risk free rate plus beta times the market's excess
    return over that same risk free rate.
    """
    b = beta(asset_returns, market_returns)
    market_mean = float(np.mean(np.asarray(market_returns, dtype=float)))
    return risk_free_rate + b * (market_mean - risk_free_rate)


def jensens_alpha(
    asset_returns: list[float],
    market_returns: list[float],
    risk_free_rate: float = 0.0,
) -> float:
    """Return Jensen's alpha for a per period return series.

    This is the average realised return minus the return the CAPM predicted
    from the asset's beta. The risk free rate must be quoted per period so it
    matches the return series.
    """
    actual = float(np.mean(np.asarray(asset_returns, dtype=float)))
    return actual - expected_return(asset_returns, market_returns, risk_free_rate)


def annualized_alpha(
    asset_returns: list[float],
    market_returns: list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Return the alpha scaled up to an annual figure.

    Alpha is a mean return per period, so annualizing it is a simple
    multiplication by the number of periods in a year.
    """
    return jensens_alpha(asset_returns, market_returns, risk_free_rate) * periods_per_year


def alpha_t_statistic(
    asset_returns: list[float],
    market_returns: list[float],
    risk_free_rate: float = 0.0,
) -> float:
    """Return the t statistic of the alpha estimate.

    The excess returns of the asset are regressed on the excess returns of
    the market. The intercept of that regression is the alpha and its
    standard error comes from the residual scatter. Roughly speaking a t
    statistic above two is the usual threshold for calling an alpha real.
    """
    a = np.asarray(asset_returns, dtype=float) - risk_free_rate
    m = np.asarray(market_returns, dtype=float) - risk_free_rate
    n = a.size
    if n < 3:
        raise ValueError("need at least three observations")
    design = np.column_stack([np.ones(n), m])
    coefficients, *_ = np.linalg.lstsq(design, a, rcond=None)
    residuals = a - design @ coefficients
    dof = n - 2
    residual_var = float(residuals @ residuals) / dof
    if residual_var == 0:
        raise ValueError("residuals have zero variance, the t statistic is undefined")
    covariance = residual_var * np.linalg.inv(design.T @ design)
    standard_error = float(np.sqrt(covariance[0, 0]))
    return float(coefficients[0]) / standard_error


if __name__ == "__main__":
    print("Jensen's Alpha")
    print("=" * 40)

    rng = np.random.default_rng(11)
    market = rng.normal(0.0004, 0.01, size=750)

    # One fund with a genuine daily edge, one that only levers the market up.
    skilled = 0.9 * market + 0.0003 + rng.normal(0.0, 0.003, size=750)
    levered = 1.6 * market + rng.normal(0.0, 0.003, size=750)

    for name, series in [("skilled", skilled), ("levered", levered)]:
        s = series.tolist()
        m = market.tolist()
        print(f"\n{name}")
        print(f"  beta             {beta(s, m):.2f}")
        print(f"  alpha (annual)   {annualized_alpha(s, m):.4f}")
        print(f"  t statistic      {alpha_t_statistic(s, m):.2f}")

    print("\nLeverage lifts returns but not alpha, only the beta term grows.")
