"""Level 2: Intermediate Python Program

This lesson builds on the fundamentals. Keep the file open in
`Documentation/Programs/level2_intermediate.py` while it runs so you can read the
comments that match the console output.

Topics covered:
- Object-oriented programming (classes, inheritance)
- Working with files (CSV/JSON)
- Basic usage of NumPy and pandas
- API call simulation and error handling

Run this file directly or through `python main.py`. Ensure dependencies are
installed first:
    pip install numpy pandas
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "sample_data"
DATA_DIR.mkdir(exist_ok=True)
SOURCE_FILE = Path(__file__).resolve()


def introduction() -> None:
    """Provide orientation so learners know where the code lives."""
    print("\n" + "#" * 60)
    print("LEVEL 2 – INTERMEDIATE PYTHON WALKTHROUGH")
    print("#" * 60)
    print("Currently executing:", SOURCE_FILE.name)
    print("Folder location:", SOURCE_FILE.parent.relative_to(Path.cwd()))
    print("We'll explore OOP, file handling, and data libraries step by step.\n")


@dataclass
class Asset:
    """Represent a financial asset with basic statistics."""

    symbol: str
    price: float
    volatility: float

    def daily_var(self, position: float) -> float:
        """Compute a basic Value at Risk estimate for a position."""
        z_score = 1.65  # 95% confidence
        return position * self.price * self.volatility * z_score


class Portfolio:
    """A collection of assets with utility methods."""

    def __init__(self) -> None:
        self.assets: Dict[str, Asset] = {}

    def add_asset(self, asset: Asset) -> None:
        self.assets[asset.symbol] = asset

    def total_value(self, positions: Dict[str, float]) -> float:
        return sum(self.assets[symbol].price * qty for symbol, qty in positions.items())

    def portfolio_var(self, positions: Dict[str, float]) -> float:
        return sum(self.assets[symbol].daily_var(qty) for symbol, qty in positions.items())


def generate_sample_csv(file_path: Path) -> None:
    """Create a small CSV file for demonstration purposes."""
    print(f"Creating a sample CSV at {file_path} with mock asset data.")
    data = {
        "symbol": ["AAPL", "MSFT", "GOOGL", "AMZN"],
        "price": [175.12, 402.15, 125.39, 185.22],
        "volatility": [0.022, 0.018, 0.025, 0.028],
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)


def read_assets_from_csv(file_path: Path) -> List[Asset]:
    """Read asset data from CSV using pandas."""
    print(f"Loading CSV data from {file_path} into Asset objects.")
    df = pd.read_csv(file_path)
    return [Asset(row.symbol, row.price, row.volatility) for row in df.itertuples()]


def numpy_statistics() -> None:
    """Showcase NumPy vectorized statistics."""
    print("\nNumPy Statistics Example")
    print("Using a NumPy array to compute mean, standard deviation, and cumulative return.")
    returns = np.array([0.01, -0.005, 0.015, -0.002, 0.007])
    print(f"Returns: {returns}")
    print(f"Mean: {returns.mean():.4f}")
    print(f"Standard Deviation: {returns.std(ddof=1):.4f}")
    print(f"Cumulative Return: {(1 + returns).prod() - 1:.4f}")


def pandas_manipulation(df: pd.DataFrame) -> None:
    """Demonstrate pandas data manipulation."""
    print("\nPandas DataFrame Summary")
    print("Display the DataFrame, descriptive statistics, then sort by volatility.")
    print(df)
    print("\nDescriptive statistics:")
    print(df.describe())

    df_sorted = df.sort_values(by="volatility", ascending=True)
    print("\nAssets sorted by volatility:")
    print(df_sorted[["symbol", "volatility"]])


def simulate_api_response() -> Dict[str, Any]:
    """Simulate an API response with JSON."""
    print("\nSimulating a fake FX rates API response (no external call made).")
    response = {
        "status": "success",
        "data": {
            "timestamp": "2025-09-27T12:00:00Z",
            "currency": "USD",
            "rates": {
                "EUR": 0.94,
                "GBP": 0.82,
                "JPY": 149.52,
            },
        },
    }
    return response


def handle_api_response(response: Dict[str, Any]) -> None:
    """Process a simulated API response."""
    print("\nAPI Response Handling")
    if response.get("status") != "success":
        raise ValueError("API call failed")

    data = response["data"]
    print(f"Timestamp: {data['timestamp']}")
    print("Exchange Rates:")
    for currency, rate in data["rates"].items():
        print(f"  1 USD = {rate} {currency}")

    # Save to JSON file
    json_file = DATA_DIR / "exchange_rates.json"
    with json_file.open("w", encoding="utf-8") as f:
        json.dump(response, f, indent=2)
    print(f"Saved rates to {json_file}")


def main() -> None:
    introduction()

    csv_file = DATA_DIR / "assets.csv"
    generate_sample_csv(csv_file)

    assets = read_assets_from_csv(csv_file)
    portfolio = Portfolio()
    for asset in assets:
        portfolio.add_asset(asset)

    positions = {"AAPL": 10, "MSFT": 5, "GOOGL": 3, "AMZN": 2}
    total = portfolio.total_value(positions)
    var = portfolio.portfolio_var(positions)
    print("\nPortfolio analytics using the Asset/Portfolio classes:")
    print(f"  Total Portfolio Value: ${total:,.2f}")
    print(f"  Portfolio Daily VaR (approx): ${var:,.2f}")

    numpy_statistics()
    pandas_manipulation(pd.read_csv(csv_file))

    response = simulate_api_response()
    handle_api_response(response)

    print("\n🎯 Level 2 complete! You've practiced OOP, file I/O, and data analysis.")


if __name__ == "__main__":
    main()
