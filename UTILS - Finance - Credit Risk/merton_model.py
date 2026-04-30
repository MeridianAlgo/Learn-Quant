"""
Merton Credit Risk Model
-------------------------
Merton (1974) treats firm equity as a call option on its assets.
Default occurs when asset value falls below debt face value at maturity.

Key outputs:
- Distance to Default (DD): Standard deviations from the default threshold
- Probability of Default (PD): Risk-neutral probability of default
- Credit Spread: Extra yield required to compensate for default risk
"""

import numpy as np
import scipy.optimize as optimize
import scipy.stats as stats


def merton_equity(
    V: float,
    F: float,
    r: float,
    sigma_V: float,
    T: float,
) -> float:
    """
    Equity value as a call option on firm assets.
    E = V * N(d1) - F * exp(-rT) * N(d2)

    Args:
        V: Asset value.
        F: Face value of debt (default threshold).
        r: Risk-free rate.
        sigma_V: Asset volatility.
        T: Debt maturity (years).

    Returns:
        float: Equity value.
    """
    d1 = (np.log(V / F) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
    d2 = d1 - sigma_V * np.sqrt(T)
    return float(V * stats.norm.cdf(d1) - F * np.exp(-r * T) * stats.norm.cdf(d2))


def merton_model(
    V: float,
    F: float,
    r: float,
    sigma_V: float,
    T: float,
) -> dict:
    """
    Full Merton model analytics.

    Args:
        V: Firm asset value.
        F: Face value of debt.
        r: Risk-free rate.
        sigma_V: Asset volatility.
        T: Debt maturity (years).

    Returns:
        dict: equity_value, debt_value, distance_to_default,
              probability_of_default, credit_spread_bps, leverage.
    """
    d1 = (np.log(V / F) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
    d2 = d1 - sigma_V * np.sqrt(T)

    equity = V * stats.norm.cdf(d1) - F * np.exp(-r * T) * stats.norm.cdf(d2)
    debt = V - equity

    # Distance to Default (physical measure, drift-adjusted)
    dd = (np.log(V / F) + (r - 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
    risk_neutral_pd = float(stats.norm.cdf(-d2))

    risky_yield = float(-np.log(debt / F) / T) if debt > 0 else np.inf
    credit_spread = max(risky_yield - r, 0.0)

    return {
        "equity_value": float(equity),
        "debt_value": float(debt),
        "distance_to_default": float(dd),
        "probability_of_default": risk_neutral_pd,
        "credit_spread_bps": float(credit_spread * 10_000),
        "leverage": float(F * np.exp(-r * T) / V),
    }


def implied_asset_value(
    E: float,
    F: float,
    r: float,
    sigma_E: float,
    T: float,
) -> dict:
    """
    Back out asset value and asset volatility from observable equity data.
    Solves the system:
      E = BS_call(V, F, r, sigma_V, T)
      sigma_E * E = N(d1) * sigma_V * V

    Args:
        E: Observed equity value (market cap).
        F: Face value of debt.
        r: Risk-free rate.
        sigma_E: Observed equity volatility.
        T: Debt maturity.

    Returns:
        dict: asset_value, asset_volatility.
    """
    def equations(params):
        V, sigma_V = params
        if V <= 0 or sigma_V <= 0:
            return [1e10, 1e10]
        d1 = (np.log(V / F) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
        d2 = d1 - sigma_V * np.sqrt(T)
        eq_val = V * stats.norm.cdf(d1) - F * np.exp(-r * T) * stats.norm.cdf(d2) - E
        eq_vol = stats.norm.cdf(d1) * sigma_V * V / E - sigma_E
        return [eq_val, eq_vol]

    V0 = E + F
    sigma_V0 = sigma_E * E / (E + F)
    sol = optimize.fsolve(equations, x0=[V0, sigma_V0])
    return {"asset_value": float(sol[0]), "asset_volatility": float(abs(sol[1]))}


if __name__ == "__main__":
    print("Merton Credit Risk Model")
    print("=" * 40)

    V, F, r, sigma_V, T = 100e6, 80e6, 0.05, 0.20, 1.0

    result = merton_model(V, F, r, sigma_V, T)
    print(f"\nFirm Assets:         ${V/1e6:.0f}M")
    print(f"Debt (Face):         ${F/1e6:.0f}M")
    print(f"Asset Volatility:    {sigma_V:.0%}")
    print(f"\nEquity Value:        ${result['equity_value']/1e6:.2f}M")
    print(f"Debt Value:          ${result['debt_value']/1e6:.2f}M")
    print(f"Distance-to-Default: {result['distance_to_default']:.3f}σ")
    print(f"Prob of Default:     {result['probability_of_default']:.2%}")
    print(f"Credit Spread:       {result['credit_spread_bps']:.1f} bps")

    print("\nCredit Spread vs Leverage:")
    print(f"{'Leverage':>10} | {'Spread (bps)':>12} | {'PD':>8}")
    print("-" * 35)
    for lev in [0.5, 0.6, 0.7, 0.8, 0.9]:
        res = merton_model(100e6, lev * 100e6, 0.05, 0.20, 1.0)
        print(f"  {lev:.1f}      | {res['credit_spread_bps']:12.1f} | {res['probability_of_default']:8.2%}")
