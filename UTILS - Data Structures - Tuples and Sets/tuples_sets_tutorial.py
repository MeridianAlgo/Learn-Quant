"""Tuples and Sets Tutorial for Financial Data.

Run with:
    python tuples_sets_tutorial.py

This module teaches immutable sequences (tuples) and unique collections (sets)
with practical finance examples.
"""

from typing import Tuple


def intro() -> None:
    """Print orientation details."""
    print("\n" + "#" * 60)
    print("PYTHON DATA STRUCTURES – TUPLES AND SETS")
    print("#" * 60)
    print("Tuples: Immutable, ordered sequences (good for fixed records)")
    print("Sets: Unordered, unique collections (good for filtering/math)\n")


def tuples_basics() -> None:
    """Demonstrate tuple basics."""
    print("=" * 60)
    print("TUPLES (Immutable Sequences)")
    print("=" * 60)

    # Example 1: Fixed trade record
    # (ticker, price, shares, date)
    trade = ("AAPL", 150.50, 100, "2023-11-01")

    print(f"Trade Record: {trade}")
    print(f"Ticker: {trade[0]}")
    print(f"Price: ${trade[1]:.2f}")

    # Tuples are immutable - this would raise an error:
    # trade[1] = 155.00  # TypeError

    # Example 2: Unpacking
    ticker, price, shares, date = trade
    print(f"\nUnpacked: {shares} shares of {ticker} on {date}")

    # Example 3: Returning multiple values from function
    def get_ohlc() -> Tuple[float, float, float, float]:
        """Return Open, High, Low, Close prices."""
        return 150.00, 155.50, 149.50, 152.25

    open_p, high_p, low_p, close_p = get_ohlc()
    print(f"\nOHLC Data: O:{open_p} H:{high_p} L:{low_p} C:{close_p}")

    # Example 4: Dictionary keys
    # Tuples can be keys because they are immutable (lists cannot)
    pair_correlations = {
        ("AAPL", "MSFT"): 0.75,
        ("AAPL", "GOOGL"): 0.65,
        ("MSFT", "GOOGL"): 0.80,
    }

    print(f"\nCorrelation AAPL-MSFT: {pair_correlations[('AAPL', 'MSFT')]}")


def sets_basics() -> None:
    """Demonstrate set basics."""
    print("\n" + "=" * 60)
    print("SETS (Unique Collections)")
    print("=" * 60)

    # Example 1: Unique tickers
    # Duplicates are automatically removed
    tickers = {"AAPL", "GOOGL", "MSFT", "TSLA"}

    print(f"Unique Tickers: {tickers}")
    print(f"Count: {len(tickers)}")

    # Example 2: Fast membership testing
    # Sets are O(1) for lookups, lists are O(n)
    watchlist = {"AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA"}

    print(f"\nIs 'AAPL' in watchlist? {'AAPL' in watchlist}")
    print(f"Is 'NFLX' in watchlist? {'NFLX' in watchlist}")

    # Example 3: Adding/Removing
    watchlist.add("META")
    watchlist.remove("TSLA")
    print(f"\nUpdated Watchlist: {watchlist}")


def set_operations() -> None:
    """Demonstrate set math operations (union, intersection, etc.)."""
    print("\n" + "=" * 60)
    print("SET OPERATIONS")
    print("=" * 60)

    # Portfolios as sets of tickers
    tech_portfolio = {"AAPL", "GOOGL", "MSFT", "NVDA", "AMD"}
    dividend_portfolio = {"MSFT", "JPM", "KO", "PEP", "CVX"}

    print(f"Tech Portfolio: {tech_portfolio}")
    print(f"Dividend Portfolio: {dividend_portfolio}")

    # Intersection (AND): Stocks in BOTH portfolios
    both = tech_portfolio.intersection(dividend_portfolio)
    # Or: tech_portfolio & dividend_portfolio
    print(f"\nIn Both (Intersection): {both}")

    # Union (OR): All unique stocks across portfolios
    all_stocks = tech_portfolio.union(dividend_portfolio)
    # Or: tech_portfolio | dividend_portfolio
    print(f"All Stocks (Union): {all_stocks}")

    # Difference: Stocks in Tech but NOT in Dividend
    tech_only = tech_portfolio.difference(dividend_portfolio)
    # Or: tech_portfolio - dividend_portfolio
    print(f"Tech Only (Difference): {tech_only}")

    # Symmetric Difference: Stocks in one OR other, but NOT both
    unique_to_each = tech_portfolio.symmetric_difference(dividend_portfolio)
    # Or: tech_portfolio ^ dividend_portfolio
    print(f"Unique to Each: {unique_to_each}")


def practical_example_deduplication() -> None:
    """Practical example: Deduplicating trade data."""
    print("\n" + "=" * 60)
    print("PRACTICAL EXAMPLE: DEDUPLICATION")
    print("=" * 60)

    # Raw trade data with duplicates (e.g., from multiple exchanges)
    raw_trades = [
        ("AAPL", 150.00, "09:30:01"),
        ("GOOGL", 2800.00, "09:30:02"),
        ("AAPL", 150.00, "09:30:01"),  # Duplicate
        ("MSFT", 300.00, "09:30:03"),
        ("GOOGL", 2800.00, "09:30:02"),  # Duplicate
    ]

    print(f"Raw Trades ({len(raw_trades)}):")
    for t in raw_trades:
        print(f"  {t}")

    # Convert to set to remove duplicates
    unique_trades = set(raw_trades)

    # Convert back to sorted list
    clean_trades = sorted(unique_trades, key=lambda x: x[2])

    print(f"\nCleaned Trades ({len(clean_trades)}):")
    for t in clean_trades:
        print(f"  {t}")


def main() -> None:
    """Run all tuple and set examples."""
    intro()
    tuples_basics()
    sets_basics()
    set_operations()
    practical_example_deduplication()
    print("\n🎉 Tuples and Sets tutorial complete!")
    print("Use tuples for fixed records and sets for unique collections.")


if __name__ == "__main__":
    main()
