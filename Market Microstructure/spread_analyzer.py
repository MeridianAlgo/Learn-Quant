"""
Spread Analysis Tools

Tools for analyzing bid-ask spreads, spread dynamics, and spread-related
trading strategies.
"""

import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class Quote:
    """Represents a market quote."""

    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    timestamp: datetime

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        return self.ask_price - self.bid_price

    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        return (self.bid_price + self.ask_price) / 2

    @property
    def spread_bps(self) -> float:
        """Calculate spread in basis points."""
        return (self.spread / self.mid_price) * 10000


class SpreadAnalyzer:
    """
    Comprehensive spread analysis tool for market microstructure research.
    """

    def __init__(self):
        self.quotes: List[Quote] = []
        self.spread_history: List[float] = []
        self.mid_price_history: List[float] = []

    def add_quote(
        self,
        bid_price: float,
        ask_price: float,
        bid_size: int,
        ask_size: int,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Add a new quote to the analyzer."""
        if timestamp is None:
            timestamp = datetime.now()

        quote = Quote(bid_price, ask_price, bid_size, ask_size, timestamp)
        self.quotes.append(quote)
        self.spread_history.append(quote.spread)
        self.mid_price_history.append(quote.mid_price)

    def get_average_spread(self, window: Optional[int] = None) -> float:
        """Calculate average spread."""
        spreads = self.spread_history[-window:] if window else self.spread_history
        return statistics.mean(spreads) if spreads else 0.0

    def get_spread_volatility(self, window: Optional[int] = None) -> float:
        """Calculate spread volatility."""
        spreads = self.spread_history[-window:] if window else self.spread_history
        return statistics.stdev(spreads) if len(spreads) > 1 else 0.0

    def get_spread_percentiles(self, window: Optional[int] = None) -> Dict[str, float]:
        """Calculate spread percentiles."""
        spreads = self.spread_history[-window:] if window else self.spread_history
        if not spreads:
            return {}

        return {
            "p25": np.percentile(spreads, 25),
            "p50": np.percentile(spreads, 50),
            "p75": np.percentile(spreads, 75),
            "p90": np.percentile(spreads, 90),
            "p95": np.percentile(spreads, 95),
        }

    def get_effective_spread(self, trade_price: float, side: str, mid_price: Optional[float] = None) -> float:
        """
        Calculate effective spread for a trade.

        Args:
            trade_price: Execution price of the trade
            side: 'buy' or 'sell'
            mid_price: Mid price at trade time (optional)

        Returns:
            Effective spread
        """
        if mid_price is None and self.quotes:
            mid_price = self.quotes[-1].mid_price
        elif mid_price is None:
            return 0.0

        if side.lower() == "buy":
            return 2 * (trade_price - mid_price)
        else:
            return 2 * (mid_price - trade_price)

    def get_realized_spread(
        self,
        trade_price: float,
        side: str,
        future_price: float,
        time_horizon: int = 300,
    ) -> float:
        """
        Calculate realized spread over a time horizon.

        Args:
            trade_price: Execution price
            side: 'buy' or 'sell'
            future_price: Price after time horizon
            time_horizon: Time in seconds (default 5 minutes)

        Returns:
            Realized spread
        """
        if side.lower() == "buy":
            return 2 * (future_price - trade_price)
        else:
            return 2 * (trade_price - future_price)

    def get_price_improvement(self, trade_price: float, side: str, quote: Optional[Quote] = None) -> float:
        """
        Calculate price improvement relative to quoted spread.

        Args:
            trade_price: Execution price
            side: 'buy' or 'sell'
            quote: Quote at trade time (optional)

        Returns:
            Price improvement (positive means better execution)
        """
        if quote is None and self.quotes:
            quote = self.quotes[-1]
        elif quote is None:
            return 0.0

        if side.lower() == "buy":
            # For buys, improvement is when price is below ask
            return quote.ask_price - trade_price
        else:
            # For sells, improvement is when price is above bid
            return trade_price - quote.bid_price

    def get_adverse_selection(
        self,
        trade_price: float,
        side: str,
        future_mid_price: float,
        current_mid_price: Optional[float] = None,
    ) -> float:
        """
        Calculate adverse selection component.

        Args:
            trade_price: Execution price
            side: 'buy' or 'sell'
            future_mid_price: Mid price after some time
            current_mid_price: Current mid price

        Returns:
            Adverse selection cost
        """
        if current_mid_price is None and self.quotes:
            current_mid_price = self.quotes[-1].mid_price
        elif current_mid_price is None:
            return 0.0

        # Adverse selection is the price movement against the trader
        if side.lower() == "buy":
            return future_mid_price - current_mid_price
        else:
            return current_mid_price - future_mid_price

    def get_spread_components(self) -> Dict[str, float]:
        """
        Decompose spread into order processing, inventory, and adverse selection.

        Returns:
            Dictionary with spread components
        """
        if len(self.quotes) < 2:
            return {"order_processing": 0, "inventory": 0, "adverse_selection": 0}

        # Simplified spread decomposition
        avg_spread = self.get_average_spread()

        # Order processing (assumed to be 40% of spread)
        order_processing = 0.4 * avg_spread

        # Inventory (assumed to be 30% of spread)
        inventory = 0.3 * avg_spread

        # Adverse selection (remaining)
        adverse_selection = avg_spread - order_processing - inventory

        return {
            "order_processing": order_processing,
            "inventory": inventory,
            "adverse_selection": adverse_selection,
            "total": avg_spread,
        }

    def get_spread_trend(self, window: int = 100) -> str:
        """
        Determine spread trend over recent window.

        Args:
            window: Number of recent quotes to analyze

        Returns:
            'widening', 'narrowing', or 'stable'
        """
        if len(self.spread_history) < window * 2:
            return "stable"

        recent = self.spread_history[-window:]
        previous = self.spread_history[-window * 2 : -window]

        recent_avg = statistics.mean(recent)
        previous_avg = statistics.mean(previous)

        change_pct = (recent_avg - previous_avg) / previous_avg

        if change_pct > 0.05:  # 5% increase
            return "widening"
        elif change_pct < -0.05:  # 5% decrease
            return "narrowing"
        else:
            return "stable"

    def get_liquidity_metrics(self) -> Dict[str, float]:
        """Calculate liquidity-related metrics."""
        if not self.quotes:
            return {}

        latest_quote = self.quotes[-1]

        # Depth at best
        depth_at_best = latest_quote.bid_size + latest_quote.ask_size

        # Spread metrics
        spread_bps = latest_quote.spread_bps
        spread_ratio = latest_quote.spread / latest_quote.mid_price

        return {
            "spread": latest_quote.spread,
            "spread_bps": spread_bps,
            "spread_ratio": spread_ratio,
            "depth_at_best": depth_at_best,
            "mid_price": latest_quote.mid_price,
        }

    def analyze_intraday_pattern(self) -> Dict[str, List[float]]:
        """
        Analyze intraday spread patterns.

        Returns:
            Dictionary with time buckets and average spreads
        """
        if not self.quotes:
            return {}

        # Group quotes by hour
        hourly_spreads = {}
        for quote in self.quotes:
            hour = quote.timestamp.hour
            if hour not in hourly_spreads:
                hourly_spreads[hour] = []
            hourly_spreads[hour].append(quote.spread)

        # Calculate average spread by hour
        hourly_avg = {}
        for hour, spreads in hourly_spreads.items():
            hourly_avg[f"{hour:02d}:00"] = statistics.mean(spreads)

        return hourly_avg

    def export_to_dataframe(self) -> pd.DataFrame:
        """Export quotes to pandas DataFrame for further analysis."""
        if not self.quotes:
            return pd.DataFrame()

        data = []
        for quote in self.quotes:
            data.append(
                {
                    "timestamp": quote.timestamp,
                    "bid_price": quote.bid_price,
                    "ask_price": quote.ask_price,
                    "bid_size": quote.bid_size,
                    "ask_size": quote.ask_size,
                    "spread": quote.spread,
                    "mid_price": quote.mid_price,
                    "spread_bps": quote.spread_bps,
                }
            )

        return pd.DataFrame(data)


def main():
    """Example usage of SpreadAnalyzer."""
    # Create analyzer
    analyzer = SpreadAnalyzer()

    # Simulate some quotes
    base_price = 100.0
    for _i in range(100):
        # Add some randomness to prices
        bid_price = base_price + np.random.normal(0, 0.01)
        ask_price = bid_price + np.random.uniform(0.01, 0.05)
        bid_size = np.random.randint(100, 1000)
        ask_size = np.random.randint(100, 1000)

        analyzer.add_quote(bid_price, ask_price, bid_size, ask_size)

    print("Spread Analysis Results")
    print("=" * 40)

    # Basic metrics
    print(f"Average Spread: ${analyzer.get_average_spread():.4f}")
    print(f"Spread Volatility: ${analyzer.get_spread_volatility():.4f}")
    print(f"Spread Trend: {analyzer.get_spread_trend()}")

    # Spread percentiles
    percentiles = analyzer.get_spread_percentiles()
    print("\nSpread Percentiles:")
    for p, value in percentiles.items():
        print(f"  {p}: ${value:.4f}")

    # Liquidity metrics
    liquidity = analyzer.get_liquidity_metrics()
    print("\nLiquidity Metrics:")
    for metric, value in liquidity.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")

    # Spread components
    components = analyzer.get_spread_components()
    print("\nSpread Components:")
    for component, value in components.items():
        print(f"  {component}: ${value:.4f}")

    # Example trade analysis
    trade_price = 100.02
    trade_side = "buy"
    effective_spread = analyzer.get_effective_spread(trade_price, trade_side)
    print("\nTrade Analysis:")
    print(f"  Trade Price: ${trade_price}")
    print(f"  Side: {trade_side}")
    print(f"  Effective Spread: ${effective_spread:.4f}")

    # Price improvement
    price_improvement = analyzer.get_price_improvement(trade_price, trade_side)
    print(f"  Price Improvement: ${price_improvement:.4f}")


if __name__ == "__main__":
    main()
