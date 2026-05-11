"""Decorators and Generators Tutorial for Financial Python.

Run with:
    python decorators_generators_tutorial.py

This module teaches:
- Decorators: Modifying function behavior (logging, timing, caching)
- Generators: Memory-efficient iteration (streaming data)
"""

import functools
import random
import time
from typing import Callable, Iterator, List


def intro() -> None:
    """Print orientation details."""
    print("\n" + "#" * 60)
    print("ADVANCED PYTHON – DECORATORS AND GENERATORS")
    print("#" * 60)
    print("Decorators: Wrappers to extend function behavior")
    print("Generators: Lazy evaluation for efficient data processing\n")


# ==========================================
# DECORATORS SECTION
# ==========================================


def decorators_demo() -> None:
    """Demonstrate various decorators."""
    print("=" * 60)
    print("DECORATORS")
    print("=" * 60)

    # 1. Timing Decorator
    def timer(func: Callable) -> Callable:
        """Print execution time of decorated function."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"⏱ {func.__name__} took {end_time - start_time:.4f} seconds")
            return result

        return wrapper

    # 2. Retry Decorator
    def retry(max_attempts: int = 3, delay: float = 1.0):
        """Retry function if it raises an exception."""

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                attempts = 0
                while attempts < max_attempts:
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        attempts += 1
                        print(f"⚠ Attempt {attempts} failed: {e}")
                        if attempts == max_attempts:
                            print("❌ All attempts failed")
                            raise
                        time.sleep(delay)

            return wrapper

        return decorator

    # 3. Caching Decorator (Memoization)
    def cache_result(func: Callable) -> Callable:
        """Cache function results based on arguments."""
        cache = {}

        @functools.wraps(func)
        def wrapper(*args):
            if args in cache:
                print(f"⚡ Returning cached result for {args}")
                return cache[args]
            result = func(*args)
            cache[args] = result
            return result

        return wrapper

    # Applying decorators
    print("\n1. Timing Decorator:")

    @timer
    def heavy_computation(n: int) -> int:
        """Simulate heavy work."""
        time.sleep(0.5)
        return n * n

    result = heavy_computation(10)
    print(f"Result: {result}")

    print("\n2. Retry Decorator:")

    @retry(max_attempts=3, delay=0.1)
    def fetch_market_data(ticker: str) -> float:
        """Simulate unreliable API."""
        if random.random() < 0.7:  # 70% chance of failure
            raise ConnectionError("API Timeout")
        return 150.00

    try:
        price = fetch_market_data("AAPL")
        print(f"Price fetched: ${price:.2f}")
    except Exception:
        print("Failed to fetch price")

    print("\n3. Caching Decorator:")

    @cache_result
    def get_historical_volatility(ticker: str, days: int) -> float:
        """Simulate expensive calculation."""
        print(f"  Computing volatility for {ticker}...")
        time.sleep(0.2)
        return 0.15

    # First call - computes
    vol1 = get_historical_volatility("AAPL", 30)

    # Second call - uses cache
    vol2 = get_historical_volatility("AAPL", 30)

    # Different args - computes
    vol3 = get_historical_volatility("GOOGL", 30)


# ==========================================
# GENERATORS SECTION
# ==========================================


def generators_demo() -> None:
    """Demonstrate generators and yield."""
    print("\n" + "=" * 60)
    print("GENERATORS")
    print("=" * 60)

    # 1. Simple Generator Function
    def count_up_to(n: int) -> Iterator[int]:
        """Yield numbers from 1 to n."""
        print("  Generator started")
        count = 1
        while count <= n:
            yield count
            count += 1
        print("  Generator finished")

    print("\n1. Simple Generator:")
    counter = count_up_to(3)
    print(f"Object: {counter}")

    print("Iterating:")
    for num in counter:
        print(f"  Got: {num}")

    # 2. Infinite Stream Generator
    def price_stream(start_price: float) -> Iterator[float]:
        """Simulate infinite stream of prices."""
        price = start_price
        while True:
            # Random walk
            change = random.uniform(-0.5, 0.5)
            price += change
            yield price

    print("\n2. Infinite Price Stream:")
    stream = price_stream(100.00)

    # Consume only 5 items from infinite stream
    for i in range(5):
        price = next(stream)
        print(f"  Tick {i + 1}: ${price:.2f}")

    # 3. Generator Expression (Memory Efficient)
    print("\n3. Generator Expression vs List Comprehension:")

    # List comprehension (creates full list in memory)
    # squares_list = [x**2 for x in range(1000000)]

    # Generator expression (lazy evaluation)
    squares_gen = (x**2 for x in range(5))

    print(f"Generator expression: {squares_gen}")
    print("Values:")
    for val in squares_gen:
        print(f"  {val}")

    # 4. Pipeline Processing
    print("\n4. Data Pipeline (Chaining Generators):")

    def read_trades():
        """Yield raw trade data."""
        trades = [
            {"symbol": "AAPL", "price": 150, "volume": 100},
            {"symbol": "GOOGL", "price": 2800, "volume": 50},
            {"symbol": "AAPL", "price": 151, "volume": 200},
            {"symbol": "MSFT", "price": 300, "volume": 150},
        ]
        yield from trades

    def filter_symbol(trades, symbol):
        """Filter trades by symbol."""
        for t in trades:
            if t["symbol"] == symbol:
                yield t

    def calculate_value(trades):
        """Calculate trade value."""
        for t in trades:
            t["value"] = t["price"] * t["volume"]
            yield t

    # Build pipeline
    raw_data = read_trades()
    aapl_trades = filter_symbol(raw_data, "AAPL")
    valued_trades = calculate_value(aapl_trades)

    # Execute pipeline
    for trade in valued_trades:
        print(f"  Processed: {trade}")


def practical_example_backtest() -> None:
    """Practical example: Event-driven backtest using generators."""
    print("\n" + "=" * 60)
    print("PRACTICAL EXAMPLE: GENERATOR BACKTEST")
    print("=" * 60)

    # 1. Market Data Generator
    def market_data_feed(tickers: List[str], periods: int):
        """Yield market data for each period."""
        for i in range(periods):
            data = {
                "date": f"2023-01-{i + 1:02d}",
                "prices": {t: random.uniform(100, 200) for t in tickers},
            }
            yield data

    # 2. Strategy Generator (Coroutine)
    def simple_strategy():
        """
        Receives market data, yields orders.
        This is a coroutine (uses yield to receive data).
        """
        print("  Strategy initialized")

        while True:
            # Receive market data
            market_data = yield

            # Simple logic: Buy if price < 150
            orders = []
            for ticker, price in market_data["prices"].items():
                if price < 120:
                    orders.append(f"BUY {ticker} @ {price:.2f}")
                elif price > 180:
                    orders.append(f"SELL {ticker} @ {price:.2f}")

            # Yield orders back to engine
            yield orders

    # 3. Backtest Engine
    print("Running Backtest:")
    tickers = ["AAPL", "MSFT"]
    feed = market_data_feed(tickers, 5)

    # Initialize strategy
    strategy = simple_strategy()
    next(strategy)  # Prime the generator (reach first yield)

    for data in feed:
        print(f"\nDate: {data['date']}")
        print(f"Prices: {data['prices']}")

        # Send data to strategy
        orders = strategy.send(data)

        # Advance strategy to next yield
        next(strategy)

        if orders:
            print(f"Orders: {orders}")
        else:
            print("No trades")


def main() -> None:
    """Run all examples."""
    intro()
    decorators_demo()
    generators_demo()
    practical_example_backtest()
    print("\n🎉 Decorators and Generators tutorial complete!")
    print("Use decorators for cross-cutting concerns and generators for efficient data pipelines.")


if __name__ == "__main__":
    main()
