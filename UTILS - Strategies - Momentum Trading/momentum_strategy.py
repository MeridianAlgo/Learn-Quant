"""Momentum Trading Strategy Simulator.

This module implements a basic Momentum strategy.
It uses price rate-of-change (ROC) and Moving Averages to determine trend direction.

Key Concepts:
- Momentum: Assets that have performed well in the past are likely to continue performing well.
- ROC (Rate of Change): Percentage change in price over a specific period.
- SMA (Simple Moving Average): Used as a trend filter.
"""

import numpy as np
import pandas as pd


def generate_synthetic_data(n_points: int = 200, seed: int = 42) -> pd.Series:
    """
    Generates a synthetic price series with trends.
    Uses a random walk with drift to simulate trending behavior.
    """
    np.random.seed(seed)

    # Returns with slight positive drift to simulate a bull market with volatility
    returns = np.random.normal(loc=0.0005, scale=0.01, size=n_points)

    # Add some "momentum" clusters (autocorrelation)
    for i in range(2, n_points):
        # If previous return was high, slight bias for next return to be high
        if returns[i - 1] > 0.01:
            returns[i] += 0.005
        elif returns[i - 1] < -0.01:
            returns[i] -= 0.005

    prices = pd.Series(np.exp(np.cumsum(returns)) * 100, name="Close")
    return prices


def calculate_momentum(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculates Rate of Change (ROC) Momentum.
    ROC = ((Price_t - Price_t-n) / Price_t-n) * 100
    """
    momentum = prices.pct_change(period) * 100
    return momentum


def calculate_sma(prices: pd.Series, window: int = 50) -> pd.Series:
    """Calculates Simple Moving Average."""
    return prices.rolling(window=window).mean()


def generate_signals(prices: pd.Series, momentum: pd.Series, sma: pd.Series, mom_threshold: float = 0.0) -> pd.Series:
    """
    Generates trading signals.
    - BUY when Momentum > 0 AND Price > SMA (Trend Following)
    - SELL/EXIT when Momentum < 0 OR Price < SMA

    Returns:
        pd.Series: 1 (Long) or 0 (Neutral)
    """
    signals = pd.Series(0, index=prices.index, name="Signal")

    # Vectorized condition
    # 1. Momentum is positive
    cond_mom_pos = momentum > mom_threshold
    # 2. Price is above trend (SMA)
    cond_trend_up = prices > sma

    signals[cond_mom_pos & cond_trend_up] = 1

    return signals


def backtest_strategy(prices: pd.Series, signals: pd.Series) -> pd.DataFrame:
    """
    Simple backtest calculating strategy returns.
    Strategy Return = Signal(t-1) * Asset Return(t)
    """
    returns = prices.pct_change()

    # Shift signals by 1 because signal at 't' is applied to return at 't+1'
    # (We enter at Close of 't', so we get return of 't+1')
    strategy_returns = signals.shift(1) * returns

    data = pd.DataFrame(
        {
            "Price": prices,
            "Signal": signals,
            "Asset_Return": returns,
            "Strategy_Return": strategy_returns,
        }
    )

    # Calculate Cumulative Returns
    data["Cumulative_Asset"] = (1 + data["Asset_Return"]).cumprod()
    data["Cumulative_Strategy"] = (1 + data["Strategy_Return"].fillna(0)).cumprod()

    return data


def main():
    print("=" * 60)
    print("MOMENTUM STRATEGY SIMULATION")
    print("=" * 60)

    # 1. Generate Data
    print("Generating synthetic price data...")
    prices = generate_synthetic_data(300)

    # 2. Calculate Indicators
    mom = calculate_momentum(prices, period=20)
    sma = calculate_sma(prices, window=50)

    # 3. Generate Signals
    signals = generate_signals(prices, mom, sma)

    # 4. Backtest
    results = backtest_strategy(prices, signals)

    # 5. Output Results
    print("\nSample Data (Tail):")
    print(results[["Price", "Signal", "Cumulative_Strategy"]].tail(10))

    total_return = results["Cumulative_Strategy"].iloc[-1] - 1
    buy_hold_return = results["Cumulative_Asset"].iloc[-1] - 1

    print("\nPerformance Summary:")
    print(f"  Total Period: {len(prices)} bars")
    print(f"  Buy & Hold Return: {buy_hold_return:.2%}")
    print(f"  Momentum Strategy Return: {total_return:.2%}")

    if total_return > buy_hold_return:
        print("  ✅ Strategy Outperformed Buy & Hold")
    else:
        print("  ⚠️ Strategy Underperformed (Trend might have been weak or chop)")


if __name__ == "__main__":
    main()
