"""
Portfolio Stress Testing
-------------------------
Apply hypothetical and historical shock scenarios to a portfolio to
estimate losses under extreme conditions.

Includes:
  - Hypothetical scenario engine (factor shocks)
  - Historical scenario replay (e.g., 2008 GFC, 2020 COVID)
  - Sensitivity (univariate stress)
  - Reverse stress test (find shock magnitude that breaches a loss threshold)
"""

from typing import Dict, List, Union

import numpy as np


def apply_scenario(
    weights: Union[list, np.ndarray],
    shocks: Union[list, np.ndarray],
    portfolio_value: float = 1.0,
) -> dict:
    """
    Apply a vector of asset return shocks to a weighted portfolio.

    Args:
        weights: Asset weights (sum to 1).
        shocks: Per-asset return shocks (e.g., -0.30 = 30% drop).
        portfolio_value: Notional value.

    Returns:
        dict: portfolio_pnl, portfolio_return, per_asset_pnl.
    """
    w = np.array(weights, dtype=float)
    s = np.array(shocks, dtype=float)
    if len(w) != len(s):
        raise ValueError("weights and shocks must have equal length")
    per_asset_ret = w * s
    port_ret = float(np.sum(per_asset_ret))
    return {
        "portfolio_return": port_ret,
        "portfolio_pnl": float(port_ret * portfolio_value),
        "per_asset_pnl": per_asset_ret * portfolio_value,
    }


HISTORICAL_SCENARIOS: Dict[str, Dict[str, float]] = {
    "2008_GFC": {"equity": -0.50, "credit": -0.30, "rates": -0.02, "commodity": -0.40, "fx": -0.15},
    "2020_COVID": {"equity": -0.34, "credit": -0.20, "rates": -0.015, "commodity": -0.55, "fx": -0.10},
    "1987_Black_Monday": {"equity": -0.225, "credit": -0.05, "rates": 0.005, "commodity": -0.10, "fx": -0.05},
    "2000_DotCom": {"equity": -0.49, "credit": -0.10, "rates": -0.03, "commodity": 0.05, "fx": -0.08},
    "2022_Inflation": {"equity": -0.20, "credit": -0.13, "rates": 0.04, "commodity": 0.30, "fx": 0.12},
}


def historical_scenario(
    weights: Dict[str, float],
    scenario: str,
    portfolio_value: float = 1.0,
) -> dict:
    """
    Apply a named historical scenario to factor-mapped weights.

    Args:
        weights: Mapping of factor name -> weight (e.g., {"equity": 0.6, "credit": 0.4}).
        scenario: Name from HISTORICAL_SCENARIOS.
        portfolio_value: Notional.

    Returns:
        dict: scenario name, portfolio P&L, per-factor P&L.
    """
    if scenario not in HISTORICAL_SCENARIOS:
        raise ValueError(f"unknown scenario '{scenario}'. Available: {list(HISTORICAL_SCENARIOS)}")
    shocks_map = HISTORICAL_SCENARIOS[scenario]
    factors = list(weights.keys())
    w = np.array([weights[f] for f in factors])
    s = np.array([shocks_map.get(f, 0.0) for f in factors])
    res = apply_scenario(w, s, portfolio_value)
    return {
        "scenario": scenario,
        "factors": factors,
        "shocks_applied": s,
        **res,
    }


def sensitivity_analysis(
    weights: Union[list, np.ndarray],
    shock_range: Union[list, np.ndarray] = None,
    asset_index: int = 0,
    portfolio_value: float = 1.0,
) -> dict:
    """
    Univariate sensitivity: vary one asset's return shock, hold others at 0.

    Args:
        weights: Portfolio weights.
        shock_range: Array of shocks to apply (default -0.5 to +0.5).
        asset_index: Index of asset to perturb.
        portfolio_value: Notional.

    Returns:
        dict: shocks, pnls.
    """
    if shock_range is None:
        shock_range = np.linspace(-0.5, 0.5, 21)
    shock_range = np.array(shock_range, dtype=float)
    w = np.array(weights, dtype=float)
    pnls = np.empty(len(shock_range))
    for i, s in enumerate(shock_range):
        shocks = np.zeros_like(w)
        shocks[asset_index] = s
        pnls[i] = float(np.sum(w * shocks)) * portfolio_value
    return {"shocks": shock_range, "pnls": pnls}


def reverse_stress_test(
    weights: Union[list, np.ndarray],
    direction: Union[list, np.ndarray],
    loss_threshold: float,
    portfolio_value: float = 1.0,
) -> float:
    """
    Find scalar shock multiplier k such that w · (k * direction) * V = -loss_threshold.

    Args:
        weights: Asset weights.
        direction: Shock direction vector (e.g., uniform negative).
        loss_threshold: Loss to breach (positive number = loss).
        portfolio_value: Notional.

    Returns:
        float: k. NaN if direction does not produce a loss.
    """
    w = np.array(weights, dtype=float)
    d = np.array(direction, dtype=float)
    sensitivity = float(np.sum(w * d)) * portfolio_value
    if sensitivity == 0.0 or sensitivity > 0:
        return float("nan")
    return float(-loss_threshold / sensitivity)


def worst_case(
    weights: Union[list, np.ndarray],
    scenario_set: List[Union[list, np.ndarray]],
    portfolio_value: float = 1.0,
) -> dict:
    """
    Compute worst (most negative) P&L across a set of scenarios.
    """
    pnls = [apply_scenario(weights, s, portfolio_value)["portfolio_pnl"] for s in scenario_set]
    idx = int(np.argmin(pnls))
    return {
        "worst_pnl": float(pnls[idx]),
        "worst_scenario_index": idx,
        "all_pnls": np.array(pnls),
    }


if __name__ == "__main__":
    weights = {"equity": 0.6, "credit": 0.3, "rates": 0.1}
    print("Historical Stress Tests (notional=$1M)")
    print("=" * 50)
    for name in HISTORICAL_SCENARIOS:
        res = historical_scenario(weights, name, portfolio_value=1_000_000)
        print(f"{name:20s} P&L: ${res['portfolio_pnl']:>14,.0f}")

    rev = reverse_stress_test([0.6, 0.4], [-1.0, -1.0], 0.20, 1.0)
    print(f"\nReverse stress (uniform down shock to lose 20%): k = {rev:.4f}")
