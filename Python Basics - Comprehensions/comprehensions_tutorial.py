"""Python Comprehensions and Functional Programming Tutorial.

Run with:
    python comprehensions_tutorial.py

Covers list, dict, and set comprehensions, generator expressions, and
functional tools (map, filter, functools.reduce, itertools.accumulate)
applied to quantitative finance workflows.
"""

import math
from functools import reduce
from itertools import accumulate
from pathlib import Path

SOURCE_FILE = Path(__file__).resolve()


def intro() -> None:
    print("\n" + "#" * 60)
    print("COMPREHENSIONS & FUNCTIONAL PYTHON")
    print("#" * 60)
    print("Executing file:", SOURCE_FILE.name)
    print("Folder location:", SOURCE_FILE.parent.relative_to(Path.cwd()))
    print("Write concise, readable, fast Python for quant workflows.\n")


def list_comprehensions() -> None:
    print("=" * 60)
    print("LIST COMPREHENSIONS")
    print("=" * 60)

    raw_tickers = ["  aapl ", "MSFT", " goog ", "tsla"]
    tickers = [t.strip().upper() for t in raw_tickers]
    print(f"Cleaned tickers:  {tickers}")

    prices = {"AAPL": 175.0, "MSFT": 420.0, "GOOG": 180.0, "TSLA": 195.0}
    over_200 = [sym for sym, px in prices.items() if px > 200]
    print(f"Stocks over $200: {over_200}")

    closing = [100.0, 102.0, 99.5, 105.0, 103.2]
    pct_returns = [
        (closing[i] - closing[i - 1]) / closing[i - 1]
        for i in range(1, len(closing))
    ]
    print(f"Daily returns:    {[round(r, 4) for r in pct_returns]}")


def dict_and_set_comprehensions() -> None:
    print("\n" + "=" * 60)
    print("DICT AND SET COMPREHENSIONS")
    print("=" * 60)

    raw = {"aapl": 175.0, "msft": 420.0, "goog": 180.0}
    normalised = {sym.upper(): round(px, 2) for sym, px in raw.items()}
    print(f"Normalised dict:       {normalised}")

    trade_log = ["AAPL", "MSFT", "AAPL", "GOOG", "TSLA", "MSFT", "AAPL"]
    unique_traded = {sym for sym in trade_log}
    print(f"Unique symbols traded: {unique_traded}")

    sectors = {"AAPL": "Tech", "MSFT": "Tech", "JPM": "Finance", "GS": "Finance"}
    sector_to_stocks = {
        sector: [sym for sym, s in sectors.items() if s == sector]
        for sector in set(sectors.values())
    }
    print(f"Sector map: {sector_to_stocks}")


def generator_expressions() -> None:
    print("\n" + "=" * 60)
    print("GENERATOR EXPRESSIONS — MEMORY EFFICIENT")
    print("=" * 60)

    # Generators compute values lazily — important for large datasets
    prices = [100.0, 102.5, 101.0, 104.0, 106.5, 105.0, 108.0]
    return_gen = (p2 / p1 - 1 for p1, p2 in zip(prices, prices[1:]))

    print("Daily returns (computed lazily):")
    for i, r in enumerate(return_gen, start=1):
        print(f"  Day {i}: {r:+.4%}")

    raw_returns = [0.01, -0.005, 0.02, -0.01, 0.015]
    total_positive = sum(r for r in raw_returns if r > 0)
    print(f"\nSum of positive returns: {total_positive:.4%}")


def map_and_filter() -> None:
    print("\n" + "=" * 60)
    print("MAP AND FILTER")
    print("=" * 60)

    prices = [100.0, 102.5, 101.0, 104.0, 106.5]
    base = prices[0]
    cumulative = list(map(lambda p: p / base - 1, prices[1:]))
    print(f"Cumulative returns from $100: {[round(r, 4) for r in cumulative]}")

    raw_signals = [0.8, -0.2, 1.5, 0.05, -1.1, 0.9, -0.05]
    strong_signals = list(filter(lambda s: abs(s) > 0.5, raw_signals))
    print(f"All signals:    {raw_signals}")
    print(f"Strong signals: {strong_signals}")

    tickers = ["AAPL", "MSFT", "GOOG", "TSLA"]
    formatted = list(map(lambda t: f"${t}", tickers))
    print(f"Formatted:      {formatted}")


def reduce_and_accumulate() -> None:
    print("\n" + "=" * 60)
    print("REDUCE AND ACCUMULATE — COMPOUNDING")
    print("=" * 60)

    daily_returns = [0.01, -0.005, 0.02, -0.01, 0.015, 0.008, -0.012]

    growth = reduce(lambda acc, r: acc * (1 + r), daily_returns, 1.0)
    total_return = growth - 1
    print(f"Daily returns:          {[round(r, 4) for r in daily_returns]}")
    print(f"Compounded total return:{total_return:.4%}")

    equity_curve = list(
        accumulate(daily_returns, lambda acc, r: acc * (1 + r), initial=10_000)
    )
    print(f"\nEquity curve (starting $10,000):")
    for day, value in enumerate(equity_curve):
        print(f"  Day {day}: ${value:,.2f}")


def nested_comprehensions_correlation() -> None:
    print("\n" + "=" * 60)
    print("NESTED COMPREHENSIONS — CORRELATION MATRIX")
    print("=" * 60)

    assets = ["AAPL", "MSFT", "GOOG"]
    returns_data = {
        "AAPL": [0.010, -0.005, 0.020, -0.010, 0.015],
        "MSFT": [0.008, -0.003, 0.018, -0.007, 0.012],
        "GOOG": [-0.002, 0.010, 0.005, 0.020, -0.008],
    }

    def pearson_r(x, y):
        n = len(x)
        mx = sum(x) / n
        my = sum(y) / n
        num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        denom = math.sqrt(
            sum((xi - mx) ** 2 for xi in x) * sum((yi - my) ** 2 for yi in y)
        )
        return num / denom if denom else 0.0

    corr_matrix = [
        [round(pearson_r(returns_data[a], returns_data[b]), 3) for b in assets]
        for a in assets
    ]

    print(f"{'':8}" + "  ".join(f"{a:>6}" for a in assets))
    for label, row in zip(assets, corr_matrix):
        print(f"{label:8}" + "  ".join(f"{v:>6.3f}" for v in row))


def main() -> None:
    intro()
    list_comprehensions()
    dict_and_set_comprehensions()
    generator_expressions()
    map_and_filter()
    reduce_and_accumulate()
    nested_comprehensions_correlation()
    print(
        "\n\U0001f389 Comprehensions tutorial complete! "
        "These patterns appear everywhere in professional quant code."
    )


if __name__ == "__main__":
    main()
