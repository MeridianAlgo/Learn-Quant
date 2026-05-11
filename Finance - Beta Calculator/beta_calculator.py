"""
Beta Calculator
Calculate beta and related metrics for portfolio and risk analysis.
"""

from typing import Dict, List

import numpy as np


def calculate_beta(asset_returns: List[float], market_returns: List[float]) -> float:
    """
    Calculate beta coefficient measuring systematic risk.

    Args:
        asset_returns: Return series for the asset
        market_returns: Return series for the market benchmark

    Returns:
        Beta coefficient
    """
    if len(asset_returns) != len(market_returns):
        raise ValueError("Return series must have same length")

    asset = np.array(asset_returns)
    market = np.array(market_returns)

    covariance = np.cov(asset, market)[0, 1]
    market_variance = np.var(market, ddof=1)

    return covariance / market_variance


def rolling_beta(asset_returns: List[float], market_returns: List[float], window: int = 60) -> List[float]:
    """
    Calculate rolling beta over a specified window.

    Args:
        asset_returns: Return series for the asset
        market_returns: Return series for the market benchmark
        window: Rolling window size

    Returns:
        List of rolling beta values
    """
    if len(asset_returns) != len(market_returns):
        raise ValueError("Return series must have same length")

    asset = np.array(asset_returns)
    market = np.array(market_returns)
    n = len(asset)

    betas = []
    for i in range(window, n + 1):
        asset_window = asset[i - window : i]
        market_window = market[i - window : i]

        covariance = np.cov(asset_window, market_window)[0, 1]
        market_variance = np.var(market_window, ddof=1)
        beta = covariance / market_variance

        betas.append(beta)

    return betas


def levered_beta(unlevered_beta: float, debt_to_equity: float, tax_rate: float = 0.21) -> float:
    """
    Calculate levered beta from unlevered beta.

    Args:
        unlevered_beta: Beta without financial leverage
        debt_to_equity: Debt-to-equity ratio
        tax_rate: Corporate tax rate

    Returns:
        Levered beta
    """
    return unlevered_beta * (1 + (1 - tax_rate) * debt_to_equity)


def unlevered_beta(levered_beta: float, debt_to_equity: float, tax_rate: float = 0.21) -> float:
    """
    Calculate unlevered beta from levered beta.

    Args:
        levered_beta: Beta with financial leverage
        debt_to_equity: Debt-to-equity ratio
        tax_rate: Corporate tax rate

    Returns:
        Unlevered beta
    """
    return levered_beta / (1 + (1 - tax_rate) * debt_to_equity)


def downside_beta(asset_returns: List[float], market_returns: List[float], threshold: float = 0.0) -> float:
    """
    Calculate downside beta, measuring systematic risk during market downturns.

    Args:
        asset_returns: Return series for the asset
        market_returns: Return series for the market benchmark
        threshold: Threshold for defining downside (typically 0)

    Returns:
        Downside beta coefficient
    """
    asset = np.array(asset_returns)
    market = np.array(market_returns)

    downside_mask = market < threshold

    if np.sum(downside_mask) < 2:
        return np.nan

    asset_downside = asset[downside_mask]
    market_downside = market[downside_mask]

    covariance = np.cov(asset_downside, market_downside)[0, 1]
    market_variance = np.var(market_downside, ddof=1)

    return covariance / market_variance


def upside_beta(asset_returns: List[float], market_returns: List[float], threshold: float = 0.0) -> float:
    """
    Calculate upside beta, measuring systematic risk during market upturns.

    Args:
        asset_returns: Return series for the asset
        market_returns: Return series for the market benchmark
        threshold: Threshold for defining upside (typically 0)

    Returns:
        Upside beta coefficient
    """
    asset = np.array(asset_returns)
    market = np.array(market_returns)

    upside_mask = market >= threshold

    if np.sum(upside_mask) < 2:
        return np.nan

    asset_upside = asset[upside_mask]
    market_upside = market[upside_mask]

    covariance = np.cov(asset_upside, market_upside)[0, 1]
    market_variance = np.var(market_upside, ddof=1)

    return covariance / market_variance


def beta_decomposition(asset_returns: List[float], market_returns: List[float]) -> Dict[str, float]:
    """
    Decompose beta into upside and downside components.

    Args:
        asset_returns: Return series for the asset
        market_returns: Return series for the market benchmark

    Returns:
        Dictionary with beta decomposition metrics
    """
    total_beta = calculate_beta(asset_returns, market_returns)
    down_beta = downside_beta(asset_returns, market_returns)
    up_beta = upside_beta(asset_returns, market_returns)

    return {
        "total_beta": total_beta,
        "downside_beta": down_beta,
        "upside_beta": up_beta,
        "beta_asymmetry": (down_beta - up_beta if not np.isnan(down_beta) and not np.isnan(up_beta) else np.nan),
    }


def adjusted_beta(historical_beta: float, adjustment_factor: float = 0.67) -> float:
    """
    Calculate adjusted beta (Blume adjustment).
    Adjusts historical beta toward market beta of 1.0.

    Args:
        historical_beta: Historical beta estimate
        adjustment_factor: Weight on historical beta (typically 0.67)

    Returns:
        Adjusted beta
    """
    return adjustment_factor * historical_beta + (1 - adjustment_factor) * 1.0


if __name__ == "__main__":
    np.random.seed(42)

    market_returns = np.random.normal(0.001, 0.02, 252)
    asset_returns = 0.3 + 1.2 * market_returns + np.random.normal(0, 0.01, 252)

    print("Beta Calculator Demo")
    print("=" * 50)

    beta = calculate_beta(asset_returns.tolist(), market_returns.tolist())
    print(f"Beta: {beta:.4f}")

    adj_beta = adjusted_beta(beta)
    print(f"Adjusted Beta: {adj_beta:.4f}")

    decomp = beta_decomposition(asset_returns.tolist(), market_returns.tolist())
    print("\nBeta Decomposition:")
    print(f"  Downside Beta: {decomp['downside_beta']:.4f}")
    print(f"  Upside Beta: {decomp['upside_beta']:.4f}")
    print(f"  Beta Asymmetry: {decomp['beta_asymmetry']:.4f}")

    levered = levered_beta(beta, debt_to_equity=0.5)
    print(f"\nLevered Beta (D/E=0.5): {levered:.4f}")
