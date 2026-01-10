"""
Dictionaries Utility - Key-Value Data Structures for Financial Analysis

This module provides comprehensive Python dictionary operations for:
- Financial data organization and lookup
- Portfolio management and tracking
- Market data storage and retrieval
- Configuration management
- Caching and performance optimization

Author: MeridianAlgo
Version: 1.1.0-Beta
"""

from typing import Dict, List, Any, Optional
from collections import defaultdict
from datetime import datetime
import json


class DictionaryOperations:
    """
    Comprehensive dictionary operations for financial applications.

    This class provides methods for:
    - Financial data management
    - Portfolio tracking
    - Market data storage
    - Configuration handling
    - Performance optimization
    """

    def __init__(self):
        """Initialize the DictionaryOperations class."""
        self.asset_database = {}
        self.price_cache = {}
        self.portfolio_registry = {}
        self.market_data = {}

    def create_asset_database(
        self, assets: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Create a structured asset database from asset data.

        Args:
            assets: List of asset dictionaries

        Returns:
            Dict[str, Dict[str, Any]]: Structured asset database
        """
        database = {}

        for asset in assets:
            ticker = asset.get("ticker")
            if ticker:
                database[ticker] = {
                    "company": asset.get("company", ""),
                    "sector": asset.get("sector", "Unknown"),
                    "industry": asset.get("industry", ""),
                    "price": asset.get("price", 0.0),
                    "market_cap": asset.get("market_cap", 0),
                    "pe_ratio": asset.get("pe_ratio", 0),
                    "dividend_yield": asset.get("dividend_yield", 0),
                    "beta": asset.get("beta", 1.0),
                    "last_updated": asset.get(
                        "last_updated", datetime.now().isoformat()
                    ),
                }

        self.asset_database = database
        return database

    def update_asset_price(
        self, ticker: str, price: float, volume: Optional[int] = None
    ) -> None:
        """
        Update asset price and related data.

        Args:
            ticker: Asset ticker symbol
            price: Current price
            volume: Trading volume (optional)
        """
        if ticker in self.asset_database:
            self.asset_database[ticker]["price"] = price
            self.asset_database[ticker]["last_updated"] = datetime.now().isoformat()

            if volume:
                self.asset_database[ticker]["volume"] = volume

    def get_asset_info(self, ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive asset information.

        Args:
            ticker: Asset ticker symbol

        Returns:
            Dict[str, Any]: Asset information
        """
        return self.asset_database.get(ticker, {})

    def search_assets(self, criteria: Dict[str, Any]) -> List[str]:
        """
        Search assets based on criteria.

        Args:
            criteria: Search criteria (e.g., {'sector': 'Technology', 'pe_ratio': {'$lt': 20}})

        Returns:
            List[str]: Matching tickers
        """
        matches = []

        for ticker, asset in self.asset_database.items():
            match = True

            for key, value in criteria.items():
                if key not in asset:
                    match = False
                    break

                asset_value = asset[key]

                # Simple comparison (can be extended for complex queries)
                if isinstance(value, dict):
                    if "$gt" in value and asset_value <= value["$gt"]:
                        match = False
                        break
                    if "$lt" in value and asset_value >= value["$lt"]:
                        match = False
                        break
                    if "$gte" in value and asset_value < value["$gte"]:
                        match = False
                        break
                    if "$lte" in value and asset_value > value["$lte"]:
                        match = False
                        break
                else:
                    if asset_value != value:
                        match = False
                        break

            if match:
                matches.append(ticker)

        return matches

    def create_portfolio_registry(
        self, portfolios: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Create a portfolio registry for multiple portfolios.

        Args:
            portfolios: List of portfolio dictionaries

        Returns:
            Dict[str, Dict[str, Any]]: Portfolio registry
        """
        registry = {}

        for portfolio in portfolios:
            portfolio_id = portfolio.get("id")
            if portfolio_id:
                registry[portfolio_id] = {
                    "name": portfolio.get("name", ""),
                    "holdings": portfolio.get("holdings", {}),
                    "cash": portfolio.get("cash", 0),
                    "total_value": portfolio.get("total_value", 0),
                    "created_date": portfolio.get(
                        "created_date", datetime.now().isoformat()
                    ),
                    "last_updated": datetime.now().isoformat(),
                }

        self.portfolio_registry = registry
        return registry

    def calculate_portfolio_value(self, portfolio_id: str) -> float:
        """
        Calculate current portfolio value.

        Args:
            portfolio_id: Portfolio identifier

        Returns:
            float: Portfolio value
        """
        if portfolio_id not in self.portfolio_registry:
            return 0.0

        portfolio = self.portfolio_registry[portfolio_id]
        total_value = portfolio["cash"]

        for ticker, shares in portfolio["holdings"].items():
            if ticker in self.asset_database:
                price = self.asset_database[ticker]["price"]
                total_value += shares * price

        portfolio["total_value"] = total_value
        portfolio["last_updated"] = datetime.now().isoformat()

        return total_value

    def get_portfolio_allocation(self, portfolio_id: str) -> Dict[str, float]:
        """
        Get portfolio allocation by asset.

        Args:
            portfolio_id: Portfolio identifier

        Returns:
            Dict[str, float]: Asset allocation percentages
        """
        if portfolio_id not in self.portfolio_registry:
            return {}

        portfolio = self.portfolio_registry[portfolio_id]
        total_value = self.calculate_portfolio_value(portfolio_id)

        if total_value == 0:
            return {}

        allocation = {}
        for ticker, shares in portfolio["holdings"].items():
            if ticker in self.asset_database:
                price = self.asset_database[ticker]["price"]
                value = shares * price
                allocation[ticker] = value / total_value

        return allocation

    def create_market_data_cache(self, market_data: Dict[str, Any]) -> None:
        """
        Create market data cache for performance.

        Args:
            market_data: Market data dictionary
        """
        self.market_data = market_data

    def get_market_data(self, ticker: str, data_type: str = "price") -> Any:
        """
        Get cached market data.

        Args:
            ticker: Asset ticker
            data_type: Type of data to retrieve

        Returns:
            Any: Requested market data
        """
        if ticker in self.market_data and data_type in self.market_data[ticker]:
            return self.market_data[ticker][data_type]
        return None

    def dictionary_comprehension_examples(self) -> None:
        """
        Demonstrate advanced dictionary comprehension techniques.
        """
        print("=== Dictionary Comprehension Examples ===")

        # Sample data
        tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        prices = [150.25, 2800.50, 350.75, 3200.00, 850.25]
        sectors = ["Technology", "Technology", "Technology", "Consumer", "Technology"]

        # Create price dictionary
        price_dict = {ticker: price for ticker, price in zip(tickers, prices)}
        print(f"Price Dictionary: {price_dict}")

        # Create sector mapping
        sector_dict = {ticker: sector for ticker, sector in zip(tickers, sectors)}
        print(f"Sector Dictionary: {sector_dict}")

        # Filter expensive stocks
        expensive_stocks = {
            ticker: price for ticker, price in price_dict.items() if price > 1000
        }
        print(f"Expensive Stocks: {expensive_stocks}")

        # Create nested market data
        market_data = {
            ticker: {
                "price": price,
                "sector": sector,
                "category": "Large Cap" if price > 500 else "Mid Cap",
            }
            for ticker, price, sector in zip(tickers, prices, sectors)
        }
        print(f"Market Data: {market_data}")

    def merge_financial_data(self, *data_dicts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple financial data dictionaries.

        Args:
            *data_dicts: Variable number of data dictionaries

        Returns:
            Dict[str, Any]: Merged dictionary
        """
        merged = {}

        for data_dict in data_dicts:
            for key, value in data_dict.items():
                if key in merged:
                    # Handle conflicts by updating with new data
                    if isinstance(merged[key], dict) and isinstance(value, dict):
                        merged[key].update(value)
                    else:
                        merged[key] = value
                else:
                    merged[key] = value

        return merged

    def export_to_json(self, filename: str, data: Dict[str, Any]) -> None:
        """
        Export dictionary data to JSON file.

        Args:
            filename: Output filename
            data: Data to export
        """
        with open(filename, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def import_from_json(self, filename: str) -> Dict[str, Any]:
        """
        Import dictionary data from JSON file.

        Args:
            filename: Input filename

        Returns:
            Dict[str, Any]: Imported data
        """
        try:
            with open(filename, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def performance_comparison(self) -> None:
        """
        Compare performance of different dictionary operations.
        """
        import time

        print("=== Performance Comparison ===")

        # Create large dataset
        large_dict = {
            f"ASSET_{i}": {
                "price": 100 + i * 0.1,
                "volume": 1000000 + i * 1000,
                "sector": "Technology" if i % 2 == 0 else "Financial",
            }
            for i in range(1000)
        }

        # Test regular dict access
        start_time = time.time()
        for i in range(10000):
            _ = large_dict.get(f"ASSET_{i % 1000}", {}).get("price", 0)
        regular_time = time.time() - start_time

        # Test defaultdict
        dd = defaultdict(lambda: {"price": 0})
        for key, value in large_dict.items():
            dd[key] = value

        start_time = time.time()
        for i in range(10000):
            _ = dd[f"ASSET_{i % 1000}"]["price"]
        defaultdict_time = time.time() - start_time

        print(f"Regular dict access: {regular_time:.4f}s")
        print(f"Defaultdict access: {defaultdict_time:.4f}s")
        print(f"Performance difference: {regular_time / defaultdict_time:.2f}x")


def main():
    """
    Main function demonstrating dictionary operations for finance.
    """
    print("=== Python Dictionaries for Financial Applications ===\n")

    # Initialize the class
    dict_ops = DictionaryOperations()

    # Demo 1: Asset Database
    print("1. Asset Database:")
    assets = [
        {
            "ticker": "AAPL",
            "company": "Apple Inc.",
            "sector": "Technology",
            "price": 150.25,
        },
        {
            "ticker": "GOOGL",
            "company": "Alphabet Inc.",
            "sector": "Technology",
            "price": 2800.50,
        },
        {
            "ticker": "MSFT",
            "company": "Microsoft Corp.",
            "sector": "Technology",
            "price": 350.75,
        },
        {
            "ticker": "JPM",
            "company": "JPMorgan Chase",
            "sector": "Financial",
            "price": 125.50,
        },
    ]

    database = dict_ops.create_asset_database(assets)
    print(f"Created database with {len(database)} assets")

    # Demo 2: Portfolio Registry
    print("\n2. Portfolio Registry:")
    portfolios = [
        {
            "id": "portfolio_001",
            "name": "Tech Portfolio",
            "holdings": {"AAPL": 100, "GOOGL": 10, "MSFT": 50},
            "cash": 50000,
        }
    ]

    registry = dict_ops.create_portfolio_registry(portfolios)
    value = dict_ops.calculate_portfolio_value("portfolio_001")
    print(f"Portfolio value: ${value:,.2f}")

    # Demo 3: Search and Filter
    print("\n3. Search and Filter:")
    tech_stocks = dict_ops.search_assets({"sector": "Technology"})
    print(f"Technology stocks: {tech_stocks}")

    # Demo 4: Dictionary Comprehensions
    dict_ops.dictionary_comprehension_examples()

    # Demo 5: Performance Comparison
    dict_ops.performance_comparison()

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
