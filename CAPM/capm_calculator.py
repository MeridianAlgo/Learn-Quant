"""
CAPM (Capital Asset Pricing Model)
----------------------------------
CAPM is the cornerstone of modern asset pricing. It says the only risk you get
*paid* for is the risk you cannot diversify away — an asset's co-movement with
the market, measured by **beta**. Everything else is noise the market expects
you to diversify.

    Expected return = risk_free + beta * (market_return - risk_free)

This module computes the CAPM expected return, **Jensen's alpha** (did a manager
beat their CAPM benchmark?), and the **security market line** used to judge
whether assets are fairly priced.
"""

from __future__ import annotations

from typing import Union

import numpy as np

ArrayLike = Union[list, np.ndarray, float]


def capm_expected_return(risk_free_rate: ArrayLike, beta: ArrayLike, market_return: ArrayLike) -> ArrayLike:
    """Expected return implied by CAPM (works on scalars or arrays).

    Args:
        risk_free_rate: Risk-free rate as a decimal (0.03 = 3%).
        beta: Asset beta — sensitivity to the market.
        market_return: Expected market return as a decimal.

    Returns:
        The CAPM expected return.
    """
    rf = np.asarray(risk_free_rate, dtype=float)
    b = np.asarray(beta, dtype=float)
    rm = np.asarray(market_return, dtype=float)
    result = rf + b * (rm - rf)
    return float(result) if result.ndim == 0 else result


def jensens_alpha(actual_return: float, risk_free_rate: float, beta: float, market_return: float) -> float:
    """Jensen's alpha: realised return minus the CAPM-required return.

    Positive alpha means the asset/manager earned more than its systematic risk
    justified — genuine skill, or an unmodelled risk factor.
    """
    required = capm_expected_return(risk_free_rate, beta, market_return)
    return float(actual_return - required)


def estimate_beta(asset_returns: ArrayLike, market_returns: ArrayLike) -> float:
    """Estimate beta as Cov(asset, market) / Var(market).

    This is the slope of the regression of asset returns on market returns —
    the same number reported on finance websites.
    """
    a = np.asarray(asset_returns, dtype=float)
    m = np.asarray(market_returns, dtype=float)
    cov = np.cov(a, m, ddof=1)
    return float(cov[0, 1] / cov[1, 1])


def security_market_line(betas: ArrayLike, risk_free_rate: float, market_return: float) -> np.ndarray:
    """The CAPM expected return for a range of betas — the SML.

    Plotting actual returns against this line shows which assets sit above
    (under-priced) or below (over-priced) their fair compensation for risk.
    """
    b = np.asarray(betas, dtype=float)
    return risk_free_rate + b * (market_return - risk_free_rate)


if __name__ == "__main__":
    rf, rm = 0.03, 0.09

    print("Capital Asset Pricing Model")
    print("=" * 40)
    for beta in (0.5, 1.0, 1.5):
        er = capm_expected_return(rf, beta, rm)
        print(f"  beta = {beta:>3}  ->  expected return = {er:.2%}")

    # A manager who returned 14% with beta 1.2.
    actual, beta = 0.14, 1.2
    alpha = jensens_alpha(actual, rf, beta, rm)
    print(f"\nManager: returned {actual:.1%} at beta {beta}")
    print(f"  CAPM required: {capm_expected_return(rf, beta, rm):.2%}")
    print(f"  Jensen's alpha: {alpha:+.2%}  ({'outperformed' if alpha > 0 else 'underperformed'})")

    rng = np.random.default_rng(0)
    mkt = rng.normal(0.0004, 0.01, 500)
    asset = 1.3 * mkt + rng.normal(0, 0.005, 500)
    print(f"\nEstimated beta from 500 returns: {estimate_beta(asset, mkt):.3f}  (true 1.30)")
