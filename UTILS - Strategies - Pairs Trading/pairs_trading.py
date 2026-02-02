"""
Pairs Trading Strategy Simulator
--------------------------------
This module implements a basic Pairs Trading strategy analysis tool.
It checks for cointegration between two synthetic asset price series and checks for stationarity
of the spread.

Key Concepts:
- Cointegration: When two non-stationary series have a linear combination that IS stationary.
- Z-Score: Measuring how far the spread is from its mean in standard deviations.
"""

from typing import Tuple

import numpy as np
import pandas as pd


def generate_synthetic_data(n_points=100) -> Tuple[pd.Series, pd.Series]:
    """
    Generates two cointegrated synthetic price series for demonstration.
    Asset Y follows Asset X plus some noise.
    """
    # Random see for reproducibility
    np.random.seed(42)

    # Asset X: Random Walk
    x_returns = np.random.normal(0, 1, n_points)
    X = pd.Series(np.cumsum(x_returns) + 100, name="Asset_X")

    # Asset Y: X + Noise
    noise = np.random.normal(0, 1, n_points)
    Y = X + 5 + noise  # Cointegrated with spread ~ 5
    Y.name = "Asset_Y"

    return X, Y


def calculate_spread(
    series_x: pd.Series, series_y: pd.Series, hedge_ratio: float = 1.0
) -> pd.Series:
    """
    Calculates the spread between two assets.
    Spread = Y - hedge_ratio * X

    Args:
        series_x (pd.Series): Price series of Asset X
        series_y (pd.Series): Price series of Asset Y
        hedge_ratio (float): Ratio to hedge X against Y

    Returns:
        pd.Series: The spread.
    """
    return series_y - (hedge_ratio * series_x)


def calculate_zscore(spread: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculates the rolling Z-Score of the spread.
    Z = (Spread - RollingMean) / RollingStd

    Args:
        spread (pd.Series): The spread series.
        window (int): Rolling window size.

    Returns:
        pd.Series: Z-Score series.
    """
    mean = spread.rolling(window=window).mean()
    std = spread.rolling(window=window).std()
    z_score = (spread - mean) / std
    return z_score


def get_signal(
    z_score: pd.Series, entry_thresh: float = 2.0, exit_thresh: float = 0.5
) -> pd.Series:
    """
    Generates trading signals based on Z-Score.
    - Short Spread (Short Y, Long X) when Z > entry_thresh
    - Long Spread (Long Y, Short X) when Z < -entry_thresh
    - Exit when |Z| < exit_thresh

    Returns:
        pd.Series: Signal (-1 for Short Spread, 1 for Long Spread, 0 for Flat)
    """
    signal = pd.Series(index=z_score.index, data=0)

    # We need to iterate because position depends on previous state (hysteresis)
    # Simple vectorization is hard for state-dependent logic, so we iterate for clarity
    current_pos = 0

    for i in range(len(z_score)):
        z = z_score.iloc[i]

        if pd.isna(z):
            continue

        if z > entry_thresh:
            current_pos = -1  # Sell Spread
        elif z < -entry_thresh:
            current_pos = 1  # Buy Spread
        elif abs(z) < exit_thresh:
            current_pos = 0  # Exit

        signal.iloc[i] = current_pos

    return signal


if __name__ == "__main__":
    # 1. Generate Data
    X, Y = generate_synthetic_data(200)

    # 2. Calculate Spread (Assuming hedge ratio 1 for this simple example)
    spread = calculate_spread(X, Y, hedge_ratio=1.0)

    # 3. Calculate Z-Score
    z_score = calculate_zscore(spread, window=30)

    # 4. Generate Signals
    signals = get_signal(z_score)

    # 5. Output
    df = pd.DataFrame(
        {"X": X, "Y": Y, "Spread": spread, "Z-Score": z_score, "Signal": signals}
    )
    print("Strategy Sample (Last 10 rows):")
    print(df.tail(10))

    # Check if we triggered any trades
    trades = df[df["Signal"] != 0].shape[0]
    print(f"\nTime periods with active specific positions: {trades}/{len(df)}")
