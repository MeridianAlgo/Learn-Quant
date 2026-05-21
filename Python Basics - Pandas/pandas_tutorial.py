"""Pandas Tutorial for Quantitative Finance.

Run with:
    python pandas_tutorial.py

Covers DataFrames, time-series resampling, rolling calculations, groupby
analysis, and signal generation — the Pandas patterns that power real quant
research pipelines.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_FILE = Path(__file__).resolve()


def intro() -> None:
    print("\n" + "#" * 60)
    print("PANDAS FOR QUANTITATIVE FINANCE")
    print("#" * 60)
    print("Executing file:", SOURCE_FILE.name)
    print("Folder location:", SOURCE_FILE.parent.relative_to(Path.cwd()))
    print("Pandas is the go-to library for financial data manipulation.\n")


def build_price_dataframe() -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame indexed by business date."""
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-02", periods=60)
    close = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.012, 60))
    high = close * (1 + np.abs(np.random.normal(0, 0.006, 60)))
    low = close * (1 - np.abs(np.random.normal(0, 0.006, 60)))
    open_ = close * (1 + np.random.normal(0, 0.005, 60))
    volume = np.random.randint(500_000, 5_000_000, 60).astype(float)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    df.index.name = "date"
    return df


def dataframe_basics(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("DATAFRAME BASICS")
    print("=" * 60)
    print(f"Shape: {df.shape}  (rows x columns)")
    print(f"\nFirst 5 rows:\n{df.head().to_string()}")
    print(f"\nColumn dtypes:\n{df.dtypes.to_string()}")
    print(f"\nClose price summary:\n{df['close'].describe().round(4).to_string()}")


def returns_and_rolling(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("RETURNS AND ROLLING WINDOWS")
    print("=" * 60)

    df = df.copy()
    df["ret"] = df["close"].pct_change()
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["sma_20"] = df["close"].rolling(20).mean()
    df["upper_bb"] = df["sma_20"] + 2 * df["close"].rolling(20).std()
    df["lower_bb"] = df["sma_20"] - 2 * df["close"].rolling(20).std()
    df["rolling_sharpe"] = (
        df["ret"].rolling(20).mean() / df["ret"].rolling(20).std()
    ) * np.sqrt(252)

    cols = ["close", "ret", "sma_20", "upper_bb", "lower_bb", "rolling_sharpe"]
    print(df[cols].tail(10).round(4).to_string())
    return df


def resampling(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("RESAMPLING — DAILY TO WEEKLY OHLC")
    print("=" * 60)

    weekly = df.resample("W").agg(
        open_price=("open", "first"),
        high_price=("high", "max"),
        low_price=("low", "min"),
        close_price=("close", "last"),
        volume=("volume", "sum"),
    )
    weekly["weekly_return"] = weekly["close_price"].pct_change()
    print(weekly.tail(8).round(4).to_string())


def groupby_by_weekday(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("GROUPBY — RETURNS BY DAY OF WEEK")
    print("=" * 60)

    df2 = df.copy()
    df2["weekday"] = df2.index.day_name()
    df2["ret"] = df2["close"].pct_change()

    stats = (
        df2.groupby("weekday")["ret"]
        .agg(avg_ret="mean", volatility="std", days="count")
    )
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    stats = stats.reindex([d for d in order if d in stats.index])
    print(stats.round(4).to_string())
    print("\nDay-of-week patterns are a useful bias check before live trading.")


def signal_generation(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("SIGNAL GENERATION — SMA CROSSOVER BACKTEST")
    print("=" * 60)

    df2 = df.copy()
    df2["ret"] = df2["close"].pct_change()
    df2["sma_5"] = df2["close"].rolling(5).mean()
    df2["sma_20"] = df2["close"].rolling(20).mean()
    df2["signal"] = np.where(df2["sma_5"] > df2["sma_20"], 1, -1)
    df2["strategy_ret"] = df2["signal"].shift(1) * df2["ret"]

    buy_hold = (1 + df2["ret"]).prod() - 1
    strategy = (1 + df2["strategy_ret"].dropna()).prod() - 1

    print("Last 5 rows with signal:")
    cols = ["close", "sma_5", "sma_20", "signal", "strategy_ret"]
    print(df2[cols].tail(5).round(4).to_string())
    print(f"\nBuy-and-hold return:    {buy_hold:.2%}")
    print(f"SMA crossover return:   {strategy:.2%}")


def main() -> None:
    intro()
    df = build_price_dataframe()
    dataframe_basics(df)
    df = returns_and_rolling(df)
    resampling(df)
    groupby_by_weekday(df)
    signal_generation(df)
    print(
        "\n\U0001f389 Pandas tutorial complete! "
        "Try applying these patterns to real data from the Market Data module."
    )


if __name__ == "__main__":
    main()
