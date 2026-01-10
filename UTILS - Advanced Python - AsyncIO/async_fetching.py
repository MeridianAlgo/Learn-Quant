"""
AsyncIO Data Fetching
---------------------
This utility demonstrates how to use Python's `asyncio` and `aiohttp` (simulated standard lib usage if aiohttp not present, or using asyncio.sleep) to perform concurrent operations.
In Quant, this is crucial for fetching data from multiple exchanges or endpoints simultaneously to reduce latency.

Note: This example uses `asyncio.sleep` to simulate network IO to avoid external dependencies like aiohttp in this barebones environment, but the pattern is identical.
"""

import asyncio
import time
import random
from typing import List, Dict


async def fetch_ticker(symbol: str) -> Dict[str, float]:
    """
    Simulates fetching a ticker from a remote API asynchronously.

    Args:
        symbol (str): The ticker symbol (e.g. 'AAPL')

    Returns:
        dict: A dictionary containing the symbol and a simulated price.
    """
    print(f"Adding request for {symbol} to event loop...")

    # Simulate network latency between 0.5 and 2.0 seconds
    delay = random.uniform(0.5, 2.0)
    await asyncio.sleep(delay)

    # Simulate a price
    price = random.uniform(100.0, 500.0)

    print(f"Finished fetching {symbol} in {delay:.2f}s")
    return {"symbol": symbol, "price": round(price, 2)}


async def fetch_all(symbols: List[str]) -> List[Dict]:
    """
    Orchestrates pulling data for all symbols concurrently.
    """
    start_time = time.time()

    # Create a list of coroutine objects
    tasks = [fetch_ticker(s) for s in symbols]

    # gather runs them concurrently
    results = await asyncio.gather(*tasks)

    end_time = time.time()
    print(f"\nTotal time for {len(symbols)} requests: {end_time - start_time:.2f}s")

    return results


def run_sync_comparison(symbols: List[str]):
    """
    Runs the same fetch logic synchronously to demonstrate the performance difference.
    """
    print("\n--- Starting Synchronous Comparison ---")
    start_time = time.time()

    results = []
    for s in symbols:
        # Blocking sleep to simulate sync request
        print(f"Fetching {s} sync...")
        time.sleep(1.0)  # Average delay
        results.append({"symbol": s, "price": 0.0})

    end_time = time.time()
    print(f"Total time for sync requests: {end_time - start_time:.2f}s")


if __name__ == "__main__":
    ticker_list = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA"]

    # Run Async
    print("--- Starting Async Fetch ---")
    # asyncio.run() is the entry point for async programs
    data = asyncio.run(fetch_all(ticker_list))
    print(f"Results: {data}")

    # Run Sync for comparison
    # run_sync_comparison(ticker_list)
