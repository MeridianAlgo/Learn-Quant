"""
Order Book Implementation

A comprehensive order book implementation for modeling market microstructure
and understanding price formation dynamics.
"""

import heapq
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple


@dataclass
class Order:
    """Represents a single order in the order book."""

    order_id: str
    price: float
    quantity: int
    side: str  # 'bid' or 'ask'
    timestamp: datetime

    def __lt__(self, other):
        # For heap ordering: bids (max-heap), asks (min-heap)
        if self.side == "bid":
            return self.price > other.price  # Reverse for max-heap
        else:
            return self.price < other.price


class OrderBook:
    """
    A limit order book implementation for market microstructure analysis.

    This class maintains bid and ask orders and provides methods for
    adding, removing, and matching orders.
    """

    def __init__(self):
        self.bids: List[Order] = []  # Max-heap for bids
        self.asks: List[Order] = []  # Min-heap for asks
        self.order_map: Dict[str, Order] = {}  # Order lookup
        self.price_levels: Dict[str, Dict[float, int]] = {"bid": {}, "ask": {}}

    def add_order(
        self,
        order_id: str,
        price: float,
        quantity: int,
        side: str,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Add a new order to the book."""
        if timestamp is None:
            timestamp = datetime.now()

        order = Order(order_id, price, quantity, side, timestamp)
        self.order_map[order_id] = order

        # Add to appropriate heap
        if side == "bid":
            heapq.heappush(self.bids, order)
        else:
            heapq.heappush(self.asks, order)

        # Update price levels
        if price not in self.price_levels[side]:
            self.price_levels[side][price] = 0
        self.price_levels[side][price] += quantity

    def remove_order(self, order_id: str) -> bool:
        """Remove an order from the book."""
        if order_id not in self.order_map:
            return False

        order = self.order_map[order_id]

        # Update price levels
        if order.price in self.price_levels[order.side]:
            self.price_levels[order.side][order.price] -= order.quantity
            if self.price_levels[order.side][order.price] <= 0:
                del self.price_levels[order.side][order.price]

        del self.order_map[order_id]
        return True

    def get_best_bid(self) -> Optional[Tuple[float, int]]:
        """Get the best bid price and quantity."""
        if not self.price_levels["bid"]:
            return None
        best_price = max(self.price_levels["bid"].keys())
        return best_price, self.price_levels["bid"][best_price]

    def get_best_ask(self) -> Optional[Tuple[float, int]]:
        """Get the best ask price and quantity."""
        if not self.price_levels["ask"]:
            return None
        best_price = min(self.price_levels["ask"].keys())
        return best_price, self.price_levels["ask"][best_price]

    def get_spread(self) -> Optional[float]:
        """Calculate the bid-ask spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()

        if best_bid is None or best_ask is None:
            return None

        return best_ask[0] - best_bid[0]

    def get_market_depth(self, levels: int = 5) -> Dict[str, List[Tuple[float, int]]]:
        """Get market depth up to specified number of levels."""
        depth = {"bid": [], "ask": []}

        # Get top bid levels
        bid_prices = sorted(self.price_levels["bid"].keys(), reverse=True)[:levels]
        for price in bid_prices:
            depth["bid"].append((price, self.price_levels["bid"][price]))

        # Get top ask levels
        ask_prices = sorted(self.price_levels["ask"].keys())[:levels]
        for price in ask_prices:
            depth["ask"].append((price, self.price_levels["ask"][price]))

        return depth

    def get_vwap(self, side: str, depth_levels: int = 10) -> Optional[float]:
        """Calculate Volume Weighted Average Price for a side."""
        if side not in self.price_levels or not self.price_levels[side]:
            return None

        total_value = 0.0
        total_volume = 0

        prices = sorted(self.price_levels[side].keys(), reverse=True if side == "bid" else True)

        for price in prices[:depth_levels]:
            volume = self.price_levels[side][price]
            total_value += price * volume
            total_volume += volume

        return total_value / total_volume if total_volume > 0 else None

    def match_orders(self) -> List[Tuple[str, str, float, int]]:
        """
        Match orders and return list of trades.
        Returns list of (buy_order_id, sell_order_id, price, quantity)
        """
        trades = []

        while self.bids and self.asks:
            # Get best bid and ask
            best_bid = self.bids[0]
            best_ask = self.asks[0]

            # Check if there's a match
            if best_bid.price >= best_ask.price:
                # Execute trade
                trade_price = best_ask.price  # Price taker pays
                trade_quantity = min(best_bid.quantity, best_ask.quantity)

                trades.append((best_bid.order_id, best_ask.order_id, trade_price, trade_quantity))

                # Update order quantities
                best_bid.quantity -= trade_quantity
                best_ask.quantity -= trade_quantity

                # Remove fully executed orders
                if best_bid.quantity == 0:
                    heapq.heappop(self.bids)
                    del self.order_map[best_bid.order_id]

                if best_ask.quantity == 0:
                    heapq.heappop(self.asks)
                    del self.order_map[best_ask.order_id]
            else:
                break  # No more matches possible

        return trades

    def get_order_flow_imbalance(self) -> Optional[float]:
        """
        Calculate order flow imbalance.
        Positive values indicate more buying pressure.
        """
        bid_volume = sum(self.price_levels["bid"].values())
        ask_volume = sum(self.price_levels["ask"].values())

        if bid_volume + ask_volume == 0:
            return None

        return (bid_volume - ask_volume) / (bid_volume + ask_volume)

    def __str__(self) -> str:
        """String representation of the order book."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        spread = self.get_spread()

        output = "Order Book Status:\n"
        if best_bid:
            output += f"Best Bid: {best_bid[0]:.2f} ({best_bid[1]} shares)\n"
        if best_ask:
            output += f"Best Ask: {best_ask[0]:.2f} ({best_ask[1]} shares)\n"
        if spread:
            output += f"Spread: {spread:.2f}\n"

        return output


def main():
    """Example usage of the OrderBook class."""
    # Create order book
    book = OrderBook()

    # Add some orders
    book.add_order("b1", 100.50, 100, "bid")
    book.add_order("b2", 100.45, 200, "bid")
    book.add_order("a1", 100.55, 150, "ask")
    book.add_order("a2", 100.60, 100, "ask")

    print(book)

    # Get market depth
    depth = book.get_market_depth(3)
    print("\nMarket Depth:")
    print(f"Bids: {depth['bid']}")
    print(f"Asks: {depth['ask']}")

    # Calculate metrics
    print(f"\nOrder Flow Imbalance: {book.get_order_flow_imbalance():.3f}")
    print(f"Bid VWAP: {book.get_vwap('bid'):.2f}")
    print(f"Ask VWAP: {book.get_vwap('ask'):.2f}")

    # Match orders
    print("\nMatching orders...")
    trades = book.match_orders()
    print(f"Trades executed: {len(trades)}")
    for trade in trades:
        print(f"Trade: {trade}")


if __name__ == "__main__":
    main()
