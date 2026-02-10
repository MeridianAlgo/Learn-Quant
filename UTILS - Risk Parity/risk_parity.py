"""
Simple Risk Parity implementation using Inverse Volatility Weighting.
"""

import numpy as np


def calculate_inverse_vol_weights(volatilities):
    """
    Calculates weights based on inverse volatility.

    Args:
        volatilities (list or np.array): List of annualized volatilities for assets.

    Returns:
        np.array: Normalized weights.
    """
    vols = np.array(volatilities)
    inv_vols = 1.0 / vols
    weights = inv_vols / np.sum(inv_vols)
    return weights


def calculate_risk_contribution(weights, cov_matrix):
    """
    Calculates the marginal risk contribution of each asset.
    """
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    marginal_contribution = np.dot(cov_matrix, weights) / portfolio_vol
    risk_contribution = weights * marginal_contribution
    return risk_contribution


if __name__ == "__main__":
    # Example assets: Stocks (high vol), Bonds (low vol), Gold (medium vol)
    asset_names = ["Stocks", "Bonds", "Gold"]
    vols = [0.18, 0.05, 0.12]

    weights = calculate_inverse_vol_weights(vols)

    print("Risk Parity (Inverse Volatility) Allocation")
    print("-" * 45)
    for name, vol, weight in zip(asset_names, vols, weights):
        print(f"{name:10} | Vol: {vol:5.1%} | Weight: {weight:6.2%}")

    print("-" * 45)
    print(f"Total Weight: {np.sum(weights):.0%}")
