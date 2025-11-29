"""Level 3: Financial Python Program

Keep this file open (`Documentation/Programs/level3_financial.py`) while it runs so you
can connect each console message with the accompanying comments.

Topics practiced in this script:
- Time value of money calculations
- Portfolio analytics and risk metrics
- Fetching market data with yfinance (optional)
- Technical indicators using pandas

Run this file with `python level3_financial.py` or select "Level 3" inside `main.py`.

Optional dependency:
    pip install yfinance
If yfinance is not installed, the script will generate synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

try:  # Optional dependency
    import yfinance as yf
except ImportError:  # pragma: no cover - yfinance optional
    yf = None

TRADING_DAYS_PER_YEAR = 252
SOURCE_FILE = Path(__file__).resolve()


def introduction() -> None:
    """Orient learners before the financial walkthrough begins."""
    print("\n" + "#" * 60)
    print("LEVEL 3 – FINANCIAL PYTHON WALKTHROUGH")
    print("#" * 60)
    print("Executing file:", SOURCE_FILE.name)
    print("Folder location:", SOURCE_FILE.parent.relative_to(Path.cwd()))
    print("We'll fetch market data (or synthesize it), build analytics, then compute indicators.\n")


def time_value_of_money(pv: float, rate: float, periods: int) -> float:
    """Compute future value of an investment using compound interest."""
    return pv * (1 + rate) ** periods


@dataclass
class PortfolioAnalytics:
    """Perform analytics on a collection of asset returns."""

    returns: pd.DataFrame  # daily returns for each asset

    def expected_returns(self) -> pd.Series:
        return self.returns.mean() * TRADING_DAYS_PER_YEAR

    def covariance_matrix(self) -> pd.DataFrame:
        return self.returns.cov() * TRADING_DAYS_PER_YEAR

    def portfolio_performance(self, weights: np.ndarray) -> Dict[str, float]:
        mean_returns = self.expected_returns()
        cov_matrix = self.covariance_matrix()

        annual_return = float(np.dot(weights, mean_returns))
        annual_volatility = float(np.sqrt(weights.T @ cov_matrix.values @ weights))
        sharpe_ratio = (annual_return - 0.02) / annual_volatility if annual_volatility > 0 else 0

        return {
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": sharpe_ratio,
        }


def generate_synthetic_prices(symbols: List[str], periods: int = 252) -> pd.DataFrame:
    """Create synthetic geometric Brownian motion price paths."""
    print("Generating synthetic price data via geometric Brownian motion.")
    rng = np.random.default_rng(42)
    prices = {}

    for symbol in symbols:
        start_price = rng.uniform(50, 200)
        daily_returns = rng.normal(loc=0.0005, scale=0.02, size=periods)
        price_path = start_price * np.exp(np.cumsum(daily_returns))
        prices[symbol] = price_path

    index = pd.date_range(end=pd.Timestamp.today(), periods=periods, freq="B")
    return pd.DataFrame(prices, index=index)


def get_market_data(symbols: List[str], period: str = "1y") -> pd.DataFrame:
    """Fetch historical market data from yfinance or fallback to synthetic data."""
    if yf is None:
        print("yfinance not installed – generating synthetic data instead.")
        return generate_synthetic_prices(symbols)

    print("Downloading data from Yahoo Finance...")
    data = yf.download(symbols, period=period, auto_adjust=True, progress=False)

    if isinstance(data, pd.DataFrame) and "Close" in data:
        close_prices = data["Close"].dropna()
    else:  # Different structure when only one ticker
        close_prices = data["Adj Close"].dropna()

    if close_prices.empty:
        print("No data returned – using synthetic data.")
        return generate_synthetic_prices(symbols)

    print("Successfully pulled closing prices from Yahoo Finance.")
    return close_prices


def compute_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate moving averages and RSI."""
    print("\nComputing technical indicators (SMA 20/50 and RSI 14) from price data.")
    returns = prices.pct_change().dropna()
    indicators = pd.DataFrame(index=prices.index)

    indicators["close"] = prices.iloc[:, 0]
    indicators["sma_20"] = indicators["close"].rolling(window=20).mean()
    indicators["sma_50"] = indicators["close"].rolling(window=50).mean()

    delta = indicators["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    roll_up = pd.Series(gain).rolling(14).mean()
    roll_down = pd.Series(loss).rolling(14).mean()
    rs = roll_up / roll_down
    indicators["rsi_14"] = 100 - (100 / (1 + rs))

    return indicators


def main() -> None:
    introduction()

    symbols = ["AAPL", "MSFT", "SPY"]
    prices = get_market_data(symbols)

    # Time value of money example
    print("\nTime Value of Money Example: future value with daily compounding.")
    future_value = time_value_of_money(pv=1000, rate=0.08 / TRADING_DAYS_PER_YEAR, periods=TRADING_DAYS_PER_YEAR)
    print(f"Future value of $1000 after one year at 8% annual rate: ${future_value:.2f}")

    # Portfolio analytics
    print("\nPortfolio Analytics: computing expected return, volatility, and Sharpe ratio.")
    returns = prices.pct_change().dropna()
    analytics = PortfolioAnalytics(returns)
    weights = np.array([0.4, 0.4, 0.2])
    performance = analytics.portfolio_performance(weights)

    print("\nPortfolio Performance (40% AAPL, 40% MSFT, 20% SPY):")
    for key, value in performance.items():
        print(f"  {key.replace('_', ' ').title()}: {value:.4f}")

    # Technical indicators
    indicators = compute_indicators(prices[[symbols[0]]])
    cleaned_indicators = indicators.dropna()

    if cleaned_indicators.empty:
        print("\nTechnical Indicators: Not enough data points to compute SMA/RSI yet.")
        print("Increase the data window or rerun after gathering more historical bars.")
    else:
        latest = cleaned_indicators.iloc[-1]
        print("\nTechnical Indicators for AAPL (latest):")
        print(f"  Price: ${latest['close']:.2f}")
        print(f"  SMA 20: ${latest['sma_20']:.2f}")
        print(f"  SMA 50: ${latest['sma_50']:.2f}")
        print(f"  RSI 14: {latest['rsi_14']:.2f}")

    print("\n📈 Level 3 complete! Continue exploring the code in the Documentation/Programs/ folder for deeper dives.")


if __name__ == "__main__":
    main()
