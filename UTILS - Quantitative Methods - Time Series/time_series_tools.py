"""Time Series Tools for Quantitative Finance Learners.

Run with:
    python time_series_tools.py

This walkthrough aligns with the intermediate/advanced lessons in
`Documentation/Programs/level3_financial.py` and `level4_advanced.py`. Keep this file open
while it runs to connect the console output with the annotated code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

SOURCE_FILE = Path(__file__).resolve()


@dataclass
class RollingMetrics:
    """Store rolling statistics for quick reporting."""

    mean: pd.Series
    volatility: pd.Series


def introduction() -> None:
    """Orient the learner before running the analysis."""
    print("\n" + "#" * 70)
    print("QUANT METHODS – TIME SERIES WALKTHROUGH")
    print("#" * 70)
    print("Executing file:", SOURCE_FILE.name)
    print("Folder location:", SOURCE_FILE.parent.relative_to(Path.cwd()))
    print("We'll simulate prices, compute rolling statistics, check stationarity, and build a simple forecast.\n")


def simulate_price_series(periods: int = 500) -> pd.Series:
    """Simulate a geometric Brownian motion price series."""
    print("Generating synthetic GBM price series to avoid external API dependencies.")
    rng = np.random.default_rng(0)
    start_price = 100
    drift = 0.0005
    volatility = 0.02

    random_shocks = rng.normal(loc=drift, scale=volatility, size=periods)
    prices = start_price * np.exp(np.cumsum(random_shocks))
    index = pd.date_range(end=pd.Timestamp.today(), periods=periods, freq="B")
    return pd.Series(prices, index=index, name="price")


def compute_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns for better additive properties."""
    print("Computing log returns from the price series.")
    return np.log(prices).diff().dropna()


def rolling_statistics(returns: pd.Series, window: int = 20) -> RollingMetrics:
    """Calculate rolling mean and volatility."""
    print(f"Calculating rolling mean/volatility with a {window}-day window.")
    mean = returns.rolling(window).mean()
    vol = returns.rolling(window).std() * np.sqrt(252)  # annualized volatility
    return RollingMetrics(mean=mean, volatility=vol)


def autocorrelation_analysis(returns: pd.Series, lags: int = 10) -> None:
    """Display autocorrelation values for the first few lags."""
    print("\nAutocorrelation check (first few lags):")
    for lag in range(1, lags + 1):
        corr = returns.autocorr(lag=lag)
        print(f"  Lag {lag}: {corr:.4f}")


def stationarity_test(returns: pd.Series) -> Tuple[float, float]:
    """Run an Augmented Dickey-Fuller test to assess stationarity."""
    print("\nRunning Augmented Dickey-Fuller test (ADF) on log returns.")
    adf_stat, p_value, *_ = adfuller(returns, autolag="AIC")
    print(f"  ADF Statistic: {adf_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("  Interpretation: Reject the null hypothesis – series appears stationary.")
    else:
        print("  Interpretation: Cannot reject the null – consider differencing or detrending.")
    return adf_stat, p_value


def ar1_forecast(returns: pd.Series) -> float:
    """Fit a simple AR(1) model using linear regression and forecast next return."""
    print("\nBuilding a quick AR(1) forecast using numpy polyfit (no heavy ARIMA model needed).")
    y = returns[1:].values
    x = returns[:-1].values
    slope, intercept = np.polyfit(x, y, 1)
    next_return = intercept + slope * returns.iloc[-1]
    print(f"  Estimated intercept: {intercept:.6f}")
    print(f"  Estimated slope (phi_1): {slope:.6f}")
    print(f"  Last observed return: {returns.iloc[-1]:.6f}")
    print(f"  Forecasted next return: {next_return:.6f}")
    return next_return


def main() -> None:
    introduction()

    prices = simulate_price_series()
    returns = compute_returns(prices)

    print("\nSummary statistics for returns:")
    print(returns.describe())

    rolling = rolling_statistics(returns)
    print("\nLatest rolling metrics:")
    print(f"  Rolling mean (last value): {rolling.mean.iloc[-1]:.6f}")
    print(f"  Rolling volatility (annualized, last value): {rolling.volatility.iloc[-1]:.4%}")

    autocorrelation_analysis(returns)
    stationarity_test(returns)
    ar1_forecast(returns)

    print("\n📊 Time series walkthrough complete! Explore the functions above to build more advanced tools.")


if __name__ == "__main__":
    main()
