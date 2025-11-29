"""Level 4: Advanced Python for Finance Program

Open `Documentation/Programs/level4_advanced.py` while this runs so you can map each
print statement to the corresponding code block and comments.

Topics practiced in this script:
- Algorithmic trading backtest with vectorized NumPy operations
- Machine learning workflow using scikit-learn
- Performance optimization hints (NumPy vs. Python loops)
- Risk management metrics (max drawdown, Sharpe ratio)

Run this file directly (`python level4_advanced.py`) or choose "Level 4" inside
`main.py`.

Install dependencies first:
    pip install numpy pandas scikit-learn
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

TRADING_DAYS = 252
np.random.seed(42)
SOURCE_FILE = Path(__file__).resolve()


def introduction() -> None:
    """Explain what the advanced walkthrough will demonstrate."""
    print("\n" + "#" * 60)
    print("LEVEL 4 – ADVANCED PYTHON WALKTHROUGH")
    print("#" * 60)
    print("Executing file:", SOURCE_FILE.name)
    print("Folder location:", SOURCE_FILE.parent.relative_to(Path.cwd()))
    print("We'll simulate a trading strategy, evaluate risk metrics, train an ML model, and showcase vectorization.\n")


@dataclass
class BacktestResult:
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float


def generate_synthetic_market(periods: int = 2 * TRADING_DAYS) -> pd.Series:
    """Simulate mean-reverting price series for strategy testing."""
    print("Generating synthetic market data to avoid external dependencies.")
    noise = np.random.normal(0, 1, periods)
    price = np.cumsum(noise) + 100
    return pd.Series(price, name="price")


def mean_reversion_strategy(prices: pd.Series, lookback: int = 20) -> pd.DataFrame:
    """Generate trading signals for a mean reversion strategy."""
    print("Calculating z-score signals for a mean reversion strategy (lookback=20).")
    df = pd.DataFrame({"price": prices})
    df["rolling_mean"] = df["price"].rolling(lookback).mean()
    df["rolling_std"] = df["price"].rolling(lookback).std()

    z_score = (df["price"] - df["rolling_mean"]) / df["rolling_std"]
    df["signal"] = 0
    df.loc[z_score > 1, "signal"] = -1  # Overbought -> short
    df.loc[z_score < -1, "signal"] = 1  # Oversold -> long
    df["signal"] = df["signal"].shift(1).fillna(0)

    df["daily_return"] = df["price"].pct_change().fillna(0)
    df["strategy_return"] = df["signal"] * df["daily_return"]
    return df


def evaluate_backtest(df: pd.DataFrame) -> BacktestResult:
    """Compute performance statistics for the backtest."""
    print("Evaluating backtest performance: cumulative return, volatility, Sharpe, drawdown.")
    cumulative_return = (1 + df["strategy_return"]).prod() - 1
    annualized_return = (1 + cumulative_return) ** (TRADING_DAYS / len(df)) - 1
    volatility = df["strategy_return"].std() * np.sqrt(TRADING_DAYS)
    sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0

    cumulative_curve = (1 + df["strategy_return"]).cumprod()
    rolling_max = cumulative_curve.cummax()
    drawdowns = (rolling_max - cumulative_curve) / rolling_max
    max_drawdown = drawdowns.max()

    return BacktestResult(
        cumulative_return=float(cumulative_return),
        annualized_return=float(annualized_return),
        annualized_volatility=float(volatility),
        sharpe_ratio=float(sharpe_ratio),
        max_drawdown=float(max_drawdown),
    )


def machine_learning_pipeline(df: pd.DataFrame) -> Tuple[RandomForestClassifier, Dict[str, str]]:
    """Build a simple ML classifier to predict next-day direction."""
    print("Training a RandomForestClassifier to predict next-day market direction.")
    features = pd.DataFrame({
        "lag_1": df["daily_return"].shift(1),
        "lag_2": df["daily_return"].shift(2),
        "volatility_5": df["daily_return"].rolling(5).std(),
        "volatility_10": df["daily_return"].rolling(10).std(),
    }).fillna(0)

    target = (df["daily_return"].shift(-1) > 0).astype(int)
    features = features.iloc[:-1]
    target = target.iloc[:-1]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    report = classification_report(y_test, predictions, output_dict=True)
    formatted_report = {
        key: f"Precision {value['precision']:.2f}, Recall {value['recall']:.2f}, F1 {value['f1-score']:.2f}"
        for key, value in report.items()
        if key in {"0", "1"}
    }

    return model, formatted_report


def vectorization_demo(size: int = 1_000_000) -> float:
    """Illustrate speed difference between Python loops and NumPy vectorization."""
    print("Comparing a Python loop sum to a NumPy vectorized sum for performance intuition.")
    python_sum = sum(i * 0.0001 for i in range(size))
    np_sum = (np.arange(size, dtype=np.float64) * 0.0001).sum()
    improvement = python_sum / np_sum if np_sum != 0 else float("inf")
    return improvement


def main() -> None:
    introduction()

    prices = generate_synthetic_market()
    strategy_df = mean_reversion_strategy(prices)
    result = evaluate_backtest(strategy_df.dropna())

    print("Backtest Performance (Mean Reversion Strategy):")
    for field, value in result.__dict__.items():
        print(f"  {field.replace('_', ' ').title()}: {value:.4f}")

    model, metrics = machine_learning_pipeline(strategy_df.dropna())
    print("\nMachine Learning Classification Report (Predicting Next-Day Direction):")
    for cls, summary in metrics.items():
        label = "Up Day" if cls == "1" else "Down Day"
        print(f"  {label}: {summary}")

    improvement = vectorization_demo()
    print(f"\nVectorization demo: NumPy sum is approximately {improvement:.2f}x comparable to Python loop result.")

    print("\n🚀 Level 4 complete! You've combined trading, ML, and performance techniques.")


if __name__ == "__main__":
    main()
