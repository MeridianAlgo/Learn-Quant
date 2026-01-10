"""Object-Oriented Programming for Financial Applications.

Run with:
    python oop_tutorial.py

This module teaches OOP concepts through building trading and portfolio classes.
"""

from typing import List, Dict, Optional
from datetime import datetime
from decimal import Decimal


def intro() -> None:
    """Print orientation details."""
    print("\n" + "#" * 60)
    print("ADVANCED PYTHON – OBJECT-ORIENTED PROGRAMMING")
    print("#" * 60)
    print("Learn classes, objects, inheritance, and encapsulation")
    print("for building robust trading systems.\n")


class Stock:
    """
    Represents a stock with ticker, price, and operations.

    This demonstrates basic class structure with attributes and methods.
    """

    def __init__(self, ticker: str, price: float, shares: int = 0):
        """
        Initialize a Stock object.

        Args:
            ticker: Stock ticker symbol
            price: Current price per share
            shares: Number of shares owned (default: 0)
        """
        self.ticker = ticker
        self.price = price
        self.shares = shares

    def get_value(self) -> float:
        """Calculate the total value of holdings."""
        return self.price * self.shares

    def update_price(self, new_price: float) -> None:
        """Update the stock price."""
        self.price = new_price

    def buy(self, shares: int) -> None:
        """Buy additional shares."""
        self.shares += shares

    def sell(self, shares: int) -> bool:
        """
        Sell shares if sufficient holdings.

        Returns:
            True if sale successful, False otherwise
        """
        if shares <= self.shares:
            self.shares -= shares
            return True
        return False

    def __str__(self) -> str:
        """String representation of the stock."""
        return f"{self.ticker}: ${self.price:.2f} × {self.shares} = ${self.get_value():,.2f}"

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"Stock(ticker='{self.ticker}', price={self.price}, shares={self.shares})"
        )


class Portfolio:
    """
    Represents an investment portfolio with multiple stocks.

    Demonstrates collections, aggregation, and portfolio-level operations.
    """

    def __init__(self, name: str, cash: float = 0.0):
        """
        Initialize a Portfolio.

        Args:
            name: Portfolio name
            cash: Starting cash balance
        """
        self.name = name
        self.cash = cash
        self.holdings: Dict[str, Stock] = {}
        self.created_at = datetime.now()

    def add_stock(self, stock: Stock) -> None:
        """Add a stock to the portfolio."""
        self.holdings[stock.ticker] = stock

    def buy_stock(self, ticker: str, price: float, shares: int) -> bool:
        """
        Buy stock shares.

        Returns:
            True if purchase successful, False if insufficient funds
        """
        cost = price * shares

        if cost > self.cash:
            return False

        self.cash -= cost

        if ticker in self.holdings:
            self.holdings[ticker].buy(shares)
            self.holdings[ticker].update_price(price)
        else:
            self.holdings[ticker] = Stock(ticker, price, shares)

        return True

    def sell_stock(self, ticker: str, shares: int) -> bool:
        """
        Sell stock shares.

        Returns:
            True if sale successful, False otherwise
        """
        if ticker not in self.holdings:
            return False

        if self.holdings[ticker].sell(shares):
            proceeds = self.holdings[ticker].price * shares
            self.cash += proceeds

            # Remove stock if no shares left
            if self.holdings[ticker].shares == 0:
                del self.holdings[ticker]

            return True

        return False

    def get_total_value(self) -> float:
        """Calculate total portfolio value (stocks + cash)."""
        stocks_value = sum(stock.get_value() for stock in self.holdings.values())
        return stocks_value + self.cash

    def get_allocation(self) -> Dict[str, float]:
        """
        Get portfolio allocation percentages.

        Returns:
            Dictionary of ticker: allocation percentage
        """
        total_value = self.get_total_value()

        if total_value == 0:
            return {}

        allocation = {}
        for ticker, stock in self.holdings.items():
            allocation[ticker] = stock.get_value() / total_value

        allocation["CASH"] = self.cash / total_value

        return allocation

    def __str__(self) -> str:
        """String representation of portfolio."""
        lines = [f"\n{'='*60}"]
        lines.append(f"Portfolio: {self.name}")
        lines.append(f"Created: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"{'='*60}")

        for stock in self.holdings.values():
            lines.append(f"  {stock}")

        lines.append(f"\n  Cash: ${self.cash:,.2f}")
        lines.append(f"  Total Value: ${self.get_total_value():,.2f}")
        lines.append("=" * 60)

        return "\n".join(lines)


class Trade:
    """
    Represents a single trade with entry and exit.

    Demonstrates properties, methods, and trade tracking.
    """

    trade_counter = 0  # Class variable to track trade IDs

    def __init__(
        self, ticker: str, entry_price: float, shares: int, direction: str = "LONG"
    ):
        """
        Initialize a Trade.

        Args:
            ticker: Stock ticker
            entry_price: Entry price
            shares: Number of shares
            direction: "LONG" or "SHORT"
        """
        Trade.trade_counter += 1
        self.trade_id = Trade.trade_counter

        self.ticker = ticker
        self.entry_price = entry_price
        self.shares = shares
        self.direction = direction.upper()
        self.entry_time = datetime.now()

        self.exit_price: Optional[float] = None
        self.exit_time: Optional[datetime] = None
        self.is_closed = False

    def close_trade(self, exit_price: float) -> None:
        """Close the trade at exit price."""
        self.exit_price = exit_price
        self.exit_time = datetime.now()
        self.is_closed = True

    @property
    def pnl(self) -> Optional[float]:
        """
        Calculate profit/loss.

        Returns:
            P&L in dollars, or None if trade not closed
        """
        if not self.is_closed or self.exit_price is None:
            return None

        if self.direction == "LONG":
            return (self.exit_price - self.entry_price) * self.shares
        else:  # SHORT
            return (self.entry_price - self.exit_price) * self.shares

    @property
    def return_pct(self) -> Optional[float]:
        """
        Calculate percentage return.

        Returns:
            Return as decimal, or None if trade not closed
        """
        if not self.is_closed or self.exit_price is None:
            return None

        if self.direction == "LONG":
            return (self.exit_price - self.entry_price) / self.entry_price
        else:  # SHORT
            return (self.entry_price - self.exit_price) / self.entry_price

    def __str__(self) -> str:
        """String representation."""
        status = "CLOSED" if self.is_closed else "OPEN"
        s = f"Trade #{self.trade_id} [{status}]: {self.direction} {self.shares} {self.ticker} @ ${self.entry_price:.2f}"

        if self.is_closed and self.exit_price and self.pnl:
            s += f" → ${self.exit_price:.2f} | P&L: ${self.pnl:.2f}"

        return s


class TradingAccount:
    """
    Full trading account with balance, trades, and performance tracking.

    Demonstrates composition and advanced OOP patterns.
    """

    def __init__(self, account_id: str, initial_balance: float):
        """Initialize trading account."""
        self.account_id = account_id
        self.balance = Decimal(str(initial_balance))
        self.initial_balance = Decimal(str(initial_balance))
        self.trades: List[Trade] = []
        self.open_positions: Dict[str, Trade] = {}

    def enter_trade(
        self, ticker: str, entry_price: float, shares: int, direction: str = "LONG"
    ) -> Optional[Trade]:
        """
        Enter a new trade.

        Returns:
            Trade object if successful, None if insufficient funds
        """
        cost = Decimal(str(entry_price * shares))

        if cost > self.balance:
            return None

        self.balance -= cost

        trade = Trade(ticker, entry_price, shares, direction)
        self.trades.append(trade)
        self.open_positions[ticker] = trade

        return trade

    def exit_trade(self, ticker: str, exit_price: float) -> bool:
        """
        Exit an open trade.

        Returns:
            True if successful, False if no open position
        """
        if ticker not in self.open_positions:
            return False

        trade = self.open_positions[ticker]
        trade.close_trade(exit_price)

        proceeds = Decimal(str(exit_price * trade.shares))
        self.balance += proceeds

        del self.open_positions[ticker]

        return True

    def get_total_pnl(self) -> float:
        """Calculate total P&L from closed trades."""
        return sum(trade.pnl for trade in self.trades if trade.pnl is not None)

    def get_win_rate(self) -> float:
        """
        Calculate win rate.

        Returns:
            Win rate as percentage (0-1)
        """
        closed_trades = [t for t in self.trades if t.is_closed and t.pnl is not None]

        if not closed_trades:
            return 0.0

        winning_trades = sum(1 for t in closed_trades if t.pnl > 0)
        return winning_trades / len(closed_trades)

    def get_current_value(self) -> float:
        """Get current account value (balance + open positions)."""
        return float(self.balance)

    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance metrics."""
        closed_trades = [t for t in self.trades if t.is_closed]

        return {
            "account_id": self.account_id,
            "current_balance": float(self.balance),
            "initial_balance": float(self.initial_balance),
            "total_pnl": self.get_total_pnl(),
            "total_trades": len(self.trades),
            "closed_trades": len(closed_trades),
            "open_positions": len(self.open_positions),
            "win_rate": self.get_win_rate(),
            "return_pct": (float(self.balance) / float(self.initial_balance)) - 1,
        }

    def __str__(self) -> str:
        """String representation."""
        perf = self.get_performance_summary()

        lines = [f"\n{'='*60}"]
        lines.append(f"Trading Account: {self.account_id}")
        lines.append(f"{'='*60}")
        lines.append(f"  Balance: ${perf['current_balance']:,.2f}")
        lines.append(f"  Total P&L: ${perf['total_pnl']:,.2f}")
        lines.append(f"  Return: {perf['return_pct']:.2%}")
        lines.append(
            f"  Trades: {perf['total_trades']} ({perf['closed_trades']} closed)"
        )
        lines.append(f"  Win Rate: {perf['win_rate']:.1%}")
        lines.append("=" * 60)

        return "\n".join(lines)


def demonstrate_classes():
    """Demonstrate basic class usage."""
    print("=" * 60)
    print("BASIC CLASSES AND OBJECTS")
    print("=" * 60)

    # Create Stock objects
    aapl = Stock("AAPL", 175.50, 50)
    googl = Stock("GOOGL", 140.25, 20)

    print("\nCreated stocks:")
    print(f"  {aapl}")
    print(f"  {googl}")

    # Stock operations
    aapl.buy(25)
    print("\nAfter buying 25 more AAPL shares:")
    print(f"  {aapl}")

    # Price update
    aapl.update_price(180.00)
    print("\nAfter AAPL price update to $180:")
    print(f"  {aapl}")


def demonstrate_portfolio():
    """Demonstrate Portfolio class."""
    print("\n" + "=" * 60)
    print("PORTFOLIO CLASS")
    print("=" * 60)

    # Create portfolio
    portfolio = Portfolio("My Portfolio", cash=10000.0)

    # Buy stocks
    portfolio.buy_stock("AAPL", 175.50, 20)
    portfolio.buy_stock("GOOGL", 140.25, 15)
    portfolio.buy_stock("MSFT", 380.75, 10)

    print(portfolio)

    # Show allocation
    allocation = portfolio.get_allocation()
    print("\nPortfolio Allocation:")
    for ticker, pct in allocation.items():
        print(f"  {ticker}: {pct:.1%}")


def demonstrate_trading_account():
    """Demonstrate TradingAccount class."""
    print("\n" + "=" * 60)
    print("TRADING ACCOUNT CLASS")
    print("=" * 60)

    # Create account
    account = TradingAccount("ACC001", 10000.0)

    # Enter trades
    trade1 = account.enter_trade("AAPL", 170.00, 25, "LONG")
    trade2 = account.enter_trade("GOOGL", 135.00, 30, "LONG")
    trade3 = account.enter_trade("TSLA", 245.00, 15, "LONG")

    print("\nOpen Trades:")
    for trade in account.open_positions.values():
        print(f"  {trade}")

    # Exit trades
    account.exit_trade("AAPL", 178.00)  # Profit
    account.exit_trade("GOOGL", 132.00)  # Loss
    account.exit_trade("TSLA", 250.00)  # Profit

    print("\nClosed Trades:")
    for trade in account.trades:
        if trade.is_closed:
            print(f"  {trade}")

    print(account)


def main() -> None:
    """Run all OOP demonstrations."""
    intro()
    demonstrate_classes()
    demonstrate_portfolio()
    demonstrate_trading_account()
    print("\n🎉 OOP tutorial complete!")
    print("Use classes to build maintainable, scalable trading systems.")


if __name__ == "__main__":
    main()
