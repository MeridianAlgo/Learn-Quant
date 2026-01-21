"""
Volatility Calculator
Calculates historical and implied volatility metrics for financial instruments.
"""

from typing import List

import numpy as np


def historical_volatility(prices: List[float], window: int = 30, annualize: bool = True) -> float:
    """
    Calculate historical volatility using log returns.

    Args:
        prices: List of historical prices
        window: Number of periods for calculation
        annualize: Whether to annualize the volatility (assumes 252 trading days)

    Returns:
        Historical volatility as a decimal
    """
    if len(prices) < window + 1:
        raise ValueError(f"Need at least {window + 1} prices for window of {window}")

    prices_array = np.array(prices[-window - 1 :])
    log_returns = np.diff(np.log(prices_array))
    volatility = np.std(log_returns, ddof=1)

    if annualize:
        volatility *= np.sqrt(252)

    return volatility


def parkinson_volatility(high: List[float], low: List[float], annualize: bool = True) -> float:
    """
    Calculate Parkinson's volatility using high-low range.
    More efficient than close-to-close volatility.

    Args:
        high: List of high prices
        low: List of low prices
        annualize: Whether to annualize the volatility

    Returns:
        Parkinson volatility as a decimal
    """
    if len(high) != len(low):
        raise ValueError("High and low arrays must have same length")

    high_array = np.array(high)
    low_array = np.array(low)

    log_hl = np.log(high_array / low_array)
    parkinson_var = np.mean(log_hl**2) / (4 * np.log(2))
    volatility = np.sqrt(parkinson_var)

    if annualize:
        volatility *= np.sqrt(252)

    return volatility


def garman_klass_volatility(
    open_prices: List[float], high: List[float], low: List[float], close: List[float], annualize: bool = True
) -> float:
    """
    Calculate Garman-Klass volatility estimator.
    Uses OHLC data for more accurate volatility estimation.

    Args:
        open_prices: List of opening prices
        high: List of high prices
        low: List of low prices
        close: List of closing prices
        annualize: Whether to annualize the volatility

    Returns:
        Garman-Klass volatility as a decimal
    """
    o = np.array(open_prices)
    h = np.array(high)
    low_array = np.array(low)
    c = np.array(close)

    log_hl = np.log(h / low_array)
    log_co = np.log(c / o)

    gk_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    volatility = np.sqrt(np.mean(gk_var))

    if annualize:
        volatility *= np.sqrt(252)

    return volatility


def ewma_volatility(returns: List[float], lambda_param: float = 0.94, annualize: bool = True) -> List[float]:
    """
    Calculate Exponentially Weighted Moving Average (EWMA) volatility.
    Used in RiskMetrics methodology.

    Args:
        returns: List of returns
        lambda_param: Decay factor (typically 0.94 for daily data)
        annualize: Whether to annualize the volatility

    Returns:
        List of EWMA volatilities
    """
    returns_array = np.array(returns)
    n = len(returns_array)

    ewma_var = np.zeros(n)
    ewma_var[0] = returns_array[0] ** 2

    for i in range(1, n):
        ewma_var[i] = lambda_param * ewma_var[i - 1] + (1 - lambda_param) * returns_array[i] ** 2

    volatility = np.sqrt(ewma_var)

    if annualize:
        volatility *= np.sqrt(252)

    return volatility.tolist()


def realized_volatility(intraday_returns: List[float], annualize: bool = True) -> float:
    """
    Calculate realized volatility from high-frequency intraday returns.

    Args:
        intraday_returns: List of intraday returns
        annualize: Whether to annualize the volatility

    Returns:
        Realized volatility as a decimal
    """
    returns_array = np.array(intraday_returns)
    realized_var = np.sum(returns_array**2)
    volatility = np.sqrt(realized_var)

    if annualize:
        volatility *= np.sqrt(252)

    return volatility


def volatility_cone(prices: List[float], windows: List[int] = None) -> dict:
    """
    Calculate volatility cone showing volatility at different time horizons.

    Args:
        prices: List of historical prices
        windows: List of window sizes to calculate

    Returns:
        Dictionary with window sizes as keys and volatility stats as values
    """
    if windows is None:
        windows = [10, 20, 30, 60, 90]
    cone = {}

    for window in windows:
        if len(prices) < window + 1:
            continue

        vols = []
        for i in range(window, len(prices)):
            vol = historical_volatility(prices[i - window : i + 1], window=window, annualize=True)
            vols.append(vol)

        if vols:
            cone[window] = {
                "min": np.min(vols),
                "max": np.max(vols),
                "mean": np.mean(vols),
                "median": np.median(vols),
                "current": vols[-1],
            }

    return cone


if __name__ == "__main__":
    np.random.seed(42)
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, 100)))

    print("Volatility Calculator Demo")
    print("=" * 50)

    hist_vol = historical_volatility(prices.tolist())
    print(f"Historical Volatility (30-day): {hist_vol:.2%}")

    high = prices * (1 + np.random.uniform(0, 0.02, len(prices)))
    low = prices * (1 - np.random.uniform(0, 0.02, len(prices)))
    park_vol = parkinson_volatility(high.tolist(), low.tolist())
    print(f"Parkinson Volatility: {park_vol:.2%}")

    returns = np.diff(np.log(prices))
    ewma_vols = ewma_volatility(returns.tolist())
    print(f"EWMA Volatility (latest): {ewma_vols[-1]:.2%}")

    cone = volatility_cone(prices.tolist())
    print("\nVolatility Cone:")
    for window, stats in cone.items():
        print(f"  {window}-day: Current={stats['current']:.2%}, Min={stats['min']:.2%}, Max={stats['max']:.2%}")
