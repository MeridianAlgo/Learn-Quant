"""
Lists Utility - Python List Operations for Financial Data

This module provides comprehensive Python list operations for:
- Financial data processing
- Portfolio management
- Transaction handling
- Algorithmic trading
- Data manipulation and analysis

Author: MeridianAlgo
Version: 1.0.0
"""

from datetime import datetime
from typing import Any, Dict, List, Optional


class ListOperations:
    """
    Comprehensive list operations for financial applications.

    This class provides methods for:
    - Financial data manipulation
    - Portfolio management
    - Transaction processing
    - Algorithmic trading
    - Performance analysis
    """

    def __init__(self):
        """Initialize the ListOperations class."""
        self.transaction_log = []

    def beginner_list_basics(self) -> None:
        """Walk beginners through core Python list concepts."""
        print("=== Beginner List Basics ===")

        watchlist = ["AAPL", "MSFT", "TSLA"]
        print(f"Initial watchlist: {watchlist}")

        watchlist.append("NVDA")
        print(f"After append('NVDA'): {watchlist}")

        watchlist.insert(1, "AMZN")
        print(f"After insert(1, 'AMZN'): {watchlist}")

        removed = watchlist.pop()
        print(f"pop() removed: {removed}")
        print(f"Watchlist now: {watchlist}")

        print("List slicing watchlist[:2]:", watchlist[:2])
        print(
            "List comprehension for name lengths:",
            [len(ticker) for ticker in watchlist],
        )

        print("Tip: Lists keep order and can hold any Python objects.\n")

    def create_portfolio_list(self, assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create a structured portfolio list from asset data.

        Args:
            assets: List of asset dictionaries with keys like 'ticker', 'shares', 'price'

        Returns:
            List[Dict[str, Any]]: Structured portfolio list
        """
        portfolio = []
        for asset in assets:
            portfolio.append(
                {
                    "ticker": asset.get("ticker"),
                    "shares": asset.get("shares", 0),
                    "price": asset.get("price", 0),
                    "value": asset.get("shares", 0) * asset.get("price", 0),
                    "weight": 0.0,  # Will be calculated
                    "sector": asset.get("sector", "Unknown"),
                    "date_added": asset.get("date_added", datetime.now().strftime("%Y-%m-%d")),
                }
            )
        return portfolio

    def calculate_portfolio_weights(self, portfolio: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculate portfolio weights based on current values.

        Args:
            portfolio: Portfolio list from create_portfolio_list

        Returns:
            List[Dict[str, Any]]: Portfolio with updated weights
        """
        total_value = sum(asset["value"] for asset in portfolio)

        for asset in portfolio:
            if total_value > 0:
                asset["weight"] = asset["value"] / total_value
            else:
                asset["weight"] = 0.0

        return portfolio

    def add_transaction(
        self,
        tx_type: str,
        ticker: str,
        shares: int,
        price: float,
        timestamp: Optional[str] = None,
    ) -> None:
        """
        Add a transaction to the transaction log.

        Args:
            tx_type: 'BUY' or 'SELL'
            ticker: Asset ticker symbol
            shares: Number of shares
            price: Transaction price
            timestamp: Transaction timestamp (auto-generated if None)
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        transaction = {
            "type": tx_type,
            "ticker": ticker,
            "shares": shares,
            "price": price,
            "value": shares * price,
            "timestamp": timestamp,
        }

        self.transaction_log.append(transaction)

    def get_transactions_by_ticker(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Get all transactions for a specific ticker.

        Args:
            ticker: Asset ticker symbol

        Returns:
            List[Dict[str, Any]]: Filtered transaction list
        """
        return [tx for tx in self.transaction_log if tx["ticker"] == ticker]

    def calculate_position_pnl(self, ticker: str, current_price: float) -> Dict[str, Any]:
        """
        Calculate profit/loss for a position.

        Args:
            ticker: Asset ticker symbol
            current_price: Current market price

        Returns:
            Dict[str, Any]: P&L analysis
        """
        ticker_transactions = self.get_transactions_by_ticker(ticker)

        total_shares = 0
        total_cost = 0.0

        for tx in ticker_transactions:
            if tx["type"] == "BUY":
                total_shares += tx["shares"]
                total_cost += tx["value"]
            elif tx["type"] == "SELL":
                total_shares -= tx["shares"]
                total_cost -= tx["value"]

        if total_shares == 0:
            return {
                "ticker": ticker,
                "total_shares": 0,
                "avg_cost": 0.0,
                "current_price": current_price,
                "market_value": 0.0,
                "total_cost": total_cost,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
            }

        avg_cost = total_cost / total_shares
        market_value = total_shares * current_price
        unrealized_pnl = market_value - total_cost

        # Calculate realized P&L (simplified)
        realized_pnl = sum(tx["value"] for tx in ticker_transactions if tx["type"] == "SELL")

        return {
            "ticker": ticker,
            "total_shares": total_shares,
            "avg_cost": avg_cost,
            "current_price": current_price,
            "market_value": market_value,
            "total_cost": total_cost,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": realized_pnl,
        }

    def sort_portfolio_by_metric(
        self, portfolio: List[Dict[str, Any]], metric: str, reverse: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Sort portfolio by a specific metric.

        Args:
            portfolio: Portfolio list
            metric: Metric to sort by ('value', 'weight', 'shares', 'price')
            reverse: Sort in descending order if True

        Returns:
            List[Dict[str, Any]]: Sorted portfolio
        """
        valid_metrics = ["value", "weight", "shares", "price", "ticker"]

        if metric not in valid_metrics:
            raise ValueError(f"Metric must be one of: {valid_metrics}")

        return sorted(portfolio, key=lambda x: x[metric], reverse=reverse)

    def filter_portfolio(
        self,
        portfolio: List[Dict[str, Any]],
        min_value: float = 0,
        max_value: float = float("inf"),
        sector: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Filter portfolio based on criteria.

        Args:
            portfolio: Portfolio list
            min_value: Minimum position value
            max_value: Maximum position value
            sector: Filter by sector

        Returns:
            List[Dict[str, Any]]: Filtered portfolio
        """
        filtered = []

        for asset in portfolio:
            if (
                asset["value"] >= min_value
                and asset["value"] <= max_value
                and (sector is None or asset["sector"] == sector)
            ):
                filtered.append(asset)

        return filtered

    def calculate_portfolio_statistics(self, portfolio: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate comprehensive portfolio statistics.

        Args:
            portfolio: Portfolio list

        Returns:
            Dict[str, Any]: Portfolio statistics
        """
        if not portfolio:
            return {}

        total_value = sum(asset["value"] for asset in portfolio)
        total_shares = sum(asset["shares"] for asset in portfolio)

        # Sector allocation
        sectors = {}
        for asset in portfolio:
            sector = asset["sector"]
            if sector in sectors:
                sectors[sector] += asset["value"]
            else:
                sectors[sector] = asset["value"]

        # Top holdings
        top_holdings = sorted(portfolio, key=lambda x: x["value"], reverse=True)[:5]

        # Concentration metrics
        top_5_value = sum(asset["value"] for asset in top_holdings)
        concentration_ratio = top_5_value / total_value if total_value > 0 else 0

        return {
            "total_value": total_value,
            "total_positions": len(portfolio),
            "avg_position_value": total_value / len(portfolio) if portfolio else 0,
            "largest_position": max((asset["value"] for asset in portfolio), default=0),
            "smallest_position": min((asset["value"] for asset in portfolio), default=0),
            "sector_allocation": sectors,
            "top_holdings": top_holdings,
            "concentration_ratio": concentration_ratio,
            "diversification_score": 1 - concentration_ratio,  # Higher is better
        }

    def moving_average_strategy(self, prices: List[float], short_window: int = 5, long_window: int = 20) -> List[str]:
        """
        Simple moving average crossover strategy.

        Args:
            prices: List of historical prices
            short_window: Short moving average window
            long_window: Long moving average window

        Returns:
            List[str]: Trading signals ('BUY', 'SELL', 'HOLD')
        """
        if len(prices) < long_window:
            return ["HOLD"] * len(prices)

        signals = []

        for i in range(len(prices)):
            if i < long_window - 1:
                signals.append("HOLD")
                continue

            # Calculate moving averages
            short_ma = sum(prices[i - short_window + 1 : i + 1]) / short_window
            long_ma = sum(prices[i - long_window + 1 : i + 1]) / long_window

            # Generate signals
            if i == 0:
                signals.append("HOLD")
            else:
                prev_short_ma = sum(prices[i - short_window : i]) / short_window
                prev_long_ma = sum(prices[i - long_window : i]) / long_window

                # Crossover signals
                if short_ma > long_ma and prev_short_ma <= prev_long_ma:
                    signals.append("BUY")
                elif short_ma < long_ma and prev_short_ma >= prev_long_ma:
                    signals.append("SELL")
                else:
                    signals.append("HOLD")

        return signals

    def rebalance_portfolio(
        self,
        current_portfolio: List[Dict[str, Any]],
        target_weights: Dict[str, float],
        current_prices: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """
        Rebalance portfolio to target weights.

        Args:
            current_portfolio: Current portfolio
            target_weights: Target weights by ticker
            current_prices: Current market prices

        Returns:
            List[Dict[str, Any]]: Rebalancing transactions
        """
        total_value = sum(asset["value"] for asset in current_portfolio)
        transactions = []

        for asset in current_portfolio:
            ticker = asset["ticker"]
            current_weight = asset["value"] / total_value if total_value > 0 else 0
            target_weight = target_weights.get(ticker, 0)

            if abs(current_weight - target_weight) > 0.01:  # 1% threshold
                target_value = total_value * target_weight
                current_value = asset["value"]
                value_diff = target_value - current_value

                if value_diff != 0:
                    shares_to_trade = value_diff / current_prices.get(ticker, asset["price"])
                    transactions.append(
                        {
                            "ticker": ticker,
                            "shares": int(shares_to_trade),
                            "type": "BUY" if shares_to_trade > 0 else "SELL",
                            "reason": "REBALANCE",
                        }
                    )

        return transactions

    def performance_analysis(self, portfolio_values: List[float]) -> Dict[str, Any]:
        """
        Analyze portfolio performance metrics.

        Args:
            portfolio_values: List of portfolio values over time

        Returns:
            Dict[str, Any]: Performance metrics
        """
        if len(portfolio_values) < 2:
            return {}

        # Calculate returns
        returns = [
            (portfolio_values[i] - portfolio_values[i - 1]) / portfolio_values[i - 1]
            for i in range(1, len(portfolio_values))
        ]

        cumulative_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]

        # Risk metrics
        avg_return = sum(returns) / len(returns)
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        volatility = variance**0.5

        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (avg_return - risk_free_rate) / volatility if volatility > 0 else 0

        # Maximum drawdown
        peak = portfolio_values[0]
        max_drawdown = 0
        current_drawdown = 0

        for value in portfolio_values:
            if value > peak:
                peak = value
                current_drawdown = 0
            else:
                current_drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, current_drawdown)

        return {
            "total_return": cumulative_return,
            "annualized_return": avg_return * 252,  # Assuming daily returns
            "volatility": volatility * np.sqrt(252),  # Annualized
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": len([r for r in returns if r > 0]) / len(returns),
            "best_day": max(returns),
            "worst_day": min(returns),
        }

    def list_comprehension_examples(self) -> None:
        """
        Demonstrate advanced list comprehension techniques for finance.
        """
        print("=== List Comprehension Examples ===")

        # Sample data
        tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        prices = [150.25, 2800.50, 350.75, 3200.00, 850.25]
        volumes = [1000000, 500000, 800000, 600000, 1200000]

        # Calculate market caps
        market_caps = [price * volume for price, volume in zip(prices, volumes)]
        print(f"Market Caps: {[f'${cap / 1e9:.1f}B' for cap in market_caps]}")

        # Filter high-volume stocks
        high_volume = [ticker for ticker, volume in zip(tickers, volumes) if volume > 700000]
        print(f"High Volume Stocks: {high_volume}")

        # Calculate percentage changes
        prev_prices = [140.00, 2700.00, 340.00, 3100.00, 800.00]
        pct_changes = [(new - old) / old * 100 for new, old in zip(prices, prev_prices)]
        print(f"Percentage Changes: {[f'{change:.2f}%' for change in pct_changes]}")

        # Create portfolio summary
        portfolio_data = list(zip(tickers, prices, volumes))
        summary = [
            {
                "ticker": ticker,
                "price": price,
                "volume": volume,
                "market_cap": price * volume,
                "category": "Large Cap" if price * volume > 1e12 else "Mid Cap",
            }
            for ticker, price, volume in portfolio_data
        ]

        print(f"Portfolio Summary: {summary[:2]}...")  # Show first 2


def main():
    """
    Main function demonstrating list operations for finance.
    """
    print("=== Python Lists for Financial Applications ===\n")

    # Initialize the class
    list_ops = ListOperations()

    # Beginner walkthrough for fundamental list concepts
    list_ops.beginner_list_basics()

    # Demo 1: Portfolio Management
    print("1. Portfolio Management:")
    assets = [
        {"ticker": "AAPL", "shares": 100, "price": 150.25, "sector": "Technology"},
        {"ticker": "GOOGL", "shares": 10, "price": 2800.50, "sector": "Technology"},
        {"ticker": "MSFT", "shares": 50, "price": 350.75, "sector": "Technology"},
        {"ticker": "JPM", "shares": 75, "price": 125.50, "sector": "Financial"},
    ]

    portfolio = list_ops.create_portfolio_list(assets)
    portfolio = list_ops.calculate_portfolio_weights(portfolio)

    print(f"Portfolio value: ${sum(asset['value'] for asset in portfolio):,.2f}")
    print(f"Top holding: {list_ops.sort_portfolio_by_metric(portfolio, 'value')[0]['ticker']}")

    # Demo 2: Transaction Processing
    print("\n2. Transaction Processing:")
    list_ops.add_transaction("BUY", "AAPL", 100, 150.25)
    list_ops.add_transaction("BUY", "GOOGL", 10, 2800.50)
    list_ops.add_transaction("SELL", "AAPL", 50, 155.00)

    aapl_pnl = list_ops.calculate_position_pnl("AAPL", 152.00)
    print(f"AAPL P&L: ${aapl_pnl['unrealized_pnl']:.2f}")

    # Demo 3: Trading Strategy
    print("\n3. Trading Strategy:")
    prices = [
        100,
        102,
        98,
        105,
        107,
        110,
        108,
        112,
        115,
        118,
        120,
        122,
        119,
        125,
        128,
        130,
        127,
        132,
        135,
        138,
    ]

    signals = list_ops.moving_average_strategy(prices, 5, 10)
    print(f"Generated {len([s for s in signals if s != 'HOLD'])} trading signals")

    # Demo 4: Performance Analysis
    print("\n4. Performance Analysis:")
    portfolio_values = [10000, 10200, 10100, 10400, 10600, 10800, 10700, 11000]
    performance = list_ops.performance_analysis(portfolio_values)

    print(f"Total return: {performance['total_return']:.2%}")
    print(f"Sharpe ratio: {performance['sharpe_ratio']:.2f}")
    print(f"Max drawdown: {performance['max_drawdown']:.2%}")

    # Demo 5: List Comprehensions
    list_ops.list_comprehension_examples()

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
