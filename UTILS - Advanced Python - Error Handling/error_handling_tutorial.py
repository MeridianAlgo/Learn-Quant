"""Error Handling and Exception Management for Trading Systems.

Run with:
    python error_handling_tutorial.py

This module teaches robust error handling for financial applications.
"""

from typing import Optional, List
import logging


def intro() -> None:
    """Print orientation details."""
    print("\n" + "#" * 60)
    print("ADVANCED PYTHON – ERROR HANDLING")
    print("#" * 60)
    print("Learn try/except, custom exceptions, logging, and")
    print("validation for building robust trading systems.\n")


def basic_try_except() -> None:
    """Demonstrate basic try/except blocks."""
    print("=" * 60)
    print("BASIC TRY/EXCEPT")
    print("=" * 60)

    # Example 1: Division by zero
    print("\nExample 1: Division by Zero")

    def calculate_return(start_price: float, end_price: float) -> float:
        """Calculate return, handling zero start price."""
        try:
            return (end_price - start_price) / start_price
        except ZeroDivisionError:
            print("  ⚠ Error: Cannot calculate return with zero start price")
            return 0.0

    print(f"  Return (100 → 110): {calculate_return(100, 110):.2%}")
    print(f"  Return (0 → 110): {calculate_return(0, 110):.2%}")

    # Example 2: Type errors
    print("\nExample 2: Type Errors")

    def calculate_position_value(price: float, shares: int) -> float:
        """Calculate position value with type checking."""
        try:
            return price * shares
        except TypeError as e:
            print(f"  ⚠ Type Error: {e}")
            return 0.0

    print(f"  Value (150.50, 100): ${calculate_position_value(150.50, 100):,.2f}")
    print(f"  Value ('invalid', 100): ${calculate_position_value('invalid', 100):,.2f}")


def multiple_exceptions() -> None:
    """Demonstrate handling multiple exception types."""
    print("\n" + "=" * 60)
    print("MULTIPLE EXCEPTION TYPES")
    print("=" * 60)

    def process_trade(ticker: str, price: str, shares: str) -> Optional[float]:
        """
        Process trade data, handling various errors.

        Returns:
            Trade value or None if errors
        """
        try:
            # Convert strings to numbers
            price_float = float(price)
            shares_int = int(shares)

            # Validate
            if price_float <= 0:
                raise ValueError("Price must be positive")
            if shares_int <= 0:
                raise ValueError("Shares must be positive")

            return price_float * shares_int

        except ValueError as e:
            print(f"  ⚠ Value Error for {ticker}: {e}")
            return None
        except TypeError as e:
            print(f"  ⚠ Type Error for {ticker}: {e}")
            return None
        except Exception as e:
            print(f"  ⚠ Unexpected Error for {ticker}: {e}")
            return None

    print("\nProcessing trades:")
    print(f"  AAPL (valid): {process_trade('AAPL', '150.50', '100')}")
    print(f"  GOOGL (negative price): {process_trade('GOOGL', '-50', '100')}")
    print(f"  MSFT (invalid shares): {process_trade('MSFT', '380', 'invalid')}")


def finally_clause() -> None:
    """Demonstrate finally clause for cleanup."""
    print("\n" + "=" * 60)
    print("FINALLY CLAUSE")
    print("=" * 60)

    def fetch_price_data(ticker: str) -> Optional[List[float]]:
        """
        Simulate fetching price data with cleanup.

        The finally block always executes, even if errors occur.
        """
        print(f"\n  → Opening connection for {ticker}...")

        try:
            # Simulate data fetching
            if ticker == "INVALID":
                raise ValueError("Invalid ticker")

            # Return mock data
            prices = [100.0, 102.0, 98.0, 105.0]
            print(f"  → Successfully fetched {len(prices)} prices")
            return prices

        except ValueError as e:
            print(f"  ⚠ Error: {e}")
            return None

        finally:
            # Always executes (cleanup code)
            print(f"  → Closing connection for {ticker}")

    fetch_price_data("AAPL")  # Success
    fetch_price_data("INVALID")  # Error, but still closes


def custom_exceptions() -> None:
    """Demonstrate custom exception classes."""
    print("\n" + "=" * 60)
    print("CUSTOM EXCEPTIONS")
    print("=" * 60)

    # Define custom exceptions
    class TradingError(Exception):
        """Base exception for trading errors."""

        pass

    class InsufficientFundsError(TradingError):
        """Raised when account has insufficient funds."""

        def __init__(self, required: float, available: float):
            self.required = required
            self.available = available
            super().__init__(
                f"Insufficient funds: need ${required:,.2f}, have ${available:,.2f}"
            )

    class InvalidTickerError(TradingError):
        """Raised when ticker symbol is invalid."""

        def __init__(self, ticker: str):
            self.ticker = ticker
            super().__init__(f"Invalid ticker symbol: {ticker}")

    # Use custom exceptions
    def execute_trade(
        ticker: str, price: float, shares: int, account_balance: float
    ) -> None:
        """Execute trade with custom error handling."""
        # Validate ticker
        valid_tickers = ["AAPL", "GOOGL", "MSFT"]
        if ticker not in valid_tickers:
            raise InvalidTickerError(ticker)

        # Check funds
        required = price * shares
        if required > account_balance:
            raise InsufficientFundsError(required, account_balance)

        print(f"  ✓ Trade executed: {shares} {ticker} @ ${price:.2f}")

    # Test trades
    print("\nExecuting trades:")

    try:
        execute_trade("AAPL", 150.00, 50, 10000.00)
    except TradingError as e:
        print(f"  ⚠ {e}")

    try:
        execute_trade("TSLA", 245.00, 100, 10000.00)
    except InvalidTickerError as e:
        print(f"  ⚠ {e}")

    try:
        execute_trade("GOOGL", 140.00, 100, 5000.00)
    except InsufficientFundsError as e:
        print(f"  ⚠ {e}")
        print(f"     Shortfall: ${e.required - e.available:,.2f}")


def else_clause() -> None:
    """Demonstrate else clause (executes if no exception)."""
    print("\n" + "=" * 60)
    print("ELSE CLAUSE")
    print("=" * 60)

    def validate_and_process_order(price: float, shares: int) -> None:
        """Process order with try/except/else/finally."""
        print(f"\n  Processing order: {shares} shares @ ${price:.2f}")

        try:
            # Validation
            if price <= 0:
                raise ValueError("Price must be positive")
            if shares <= 0:
                raise ValueError("Shares must be positive")
            if shares > 10000:
                raise ValueError("Order size too large")

        except ValueError as e:
            # Handle errors
            print(f"  ⚠ Validation failed: {e}")

        else:
            # Only executes if no exception
            print("  ✓ Validation passed")
            total_cost = price * shares
            print(f"  → Order total: ${total_cost:,.2f}")

        finally:
            # Always executes
            print("  → Order processing complete")

    validate_and_process_order(150.50, 100)  # Valid
    validate_and_process_order(-50.00, 100)  # Invalid price
    validate_and_process_order(150.50, 15000)  # Too large


def raising_exceptions() -> None:
    """Demonstrate raising exceptions."""
    print("\n" + "=" * 60)
    print("RAISING EXCEPTIONS")
    print("=" * 60)

    def validate_portfolio_allocation(allocations: dict) -> None:
        """
        Validate portfolio allocations.

        Raises:
            ValueError: If allocations are invalid
        """
        total = sum(allocations.values())

        # Check if allocations sum to 1.0 (100%)
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Allocations must sum to 100%, got {total:.1%}")

        # Check for negative allocations
        for ticker, allocation in allocations.items():
            if allocation < 0:
                raise ValueError(f"Negative allocation for {ticker}: {allocation:.1%}")

        print("  ✓ Portfolio allocation valid")

    print("\nValidating allocations:")

    # Valid allocation
    try:
        validate_portfolio_allocation({"AAPL": 0.30, "GOOGL": 0.30, "MSFT": 0.40})
    except ValueError as e:
        print(f"  ⚠ {e}")

    # Invalid - doesn't sum to 100%
    try:
        validate_portfolio_allocation({"AAPL": 0.30, "GOOGL": 0.30, "MSFT": 0.30})
    except ValueError as e:
        print(f"  ⚠ {e}")

    # Invalid - negative allocation
    try:
        validate_portfolio_allocation({"AAPL": 0.50, "GOOGL": 0.70, "MSFT": -0.20})
    except ValueError as e:
        print(f"  ⚠ {e}")


def context_managers() -> None:
    """Demonstrate context managers for resource management."""
    print("\n" + "=" * 60)
    print("CONTEXT MANAGERS (with statement)")
    print("=" * 60)

    class TradingSession:
        """Context manager for trading sessions."""

        def __init__(self, account_id: str):
            self.account_id = account_id

        def __enter__(self):
            print(f"\n  → Opening trading session for {self.account_id}")
            # Setup code here
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            print(f"  → Closing trading session for {self.account_id}")
            # Cleanup code here

            # Return False to propagate exceptions, True to suppress
            return False

        def execute_trade(self, ticker: str, shares: int):
            print(f"  → Executing: {shares} {ticker}")

    # Use context manager
    with TradingSession("ACC001") as session:
        session.execute_trade("AAPL", 100)
        session.execute_trade("GOOGL", 50)
    # Session automatically closed


def assertion_checking() -> None:
    """Demonstrate assertions for debugging."""
    print("\n" + "=" * 60)
    print("ASSERTIONS")
    print("=" * 60)

    def calculate_sharpe_ratio(
        returns: List[float], risk_free_rate: float = 0.02
    ) -> float:
        """
        Calculate Sharpe ratio with assertions.

        Assertions help catch bugs during development.
        """
        # Assertions for debugging (removed in production with -O flag)
        assert len(returns) > 0, "Returns list cannot be empty"
        assert all(
            isinstance(r, (int, float)) for r in returns
        ), "Returns must be numbers"
        assert 0 <= risk_free_rate <= 1, "Risk-free rate must be between 0 and 1"

        avg_return = sum(returns) / len(returns)
        std_dev = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5

        if std_dev == 0:
            return 0.0

        return (avg_return - risk_free_rate) / std_dev

    print("\nCalculating Sharpe ratios:")

    # Valid
    returns = [0.10, 0.05, 0.12, 0.08, 0.15]
    sharpe = calculate_sharpe_ratio(returns)
    print(f"  Sharpe Ratio: {sharpe:.3f}")

    # This would trigger assertion:
    # calculate_sharpe_ratio([])  # Empty list


def logging_errors() -> None:
    """Demonstrate logging for production error tracking."""
    print("\n" + "=" * 60)
    print("LOGGING ERRORS")
    print("=" * 60)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(__name__)

    def process_trades(trades: List[dict]) -> None:
        """Process trades with logging."""
        logger.info(f"Processing {len(trades)} trades")

        successful = 0
        failed = 0

        for trade in trades:
            try:
                ticker = trade["ticker"]
                price = float(trade["price"])
                shares = int(trade["shares"])

                value = price * shares
                logger.debug(f"Processed {ticker}: ${value:,.2f}")
                successful += 1

            except KeyError as e:
                logger.error(f"Missing field in trade: {e}")
                failed += 1
            except (ValueError, TypeError) as e:
                logger.error(f"Invalid trade data: {e}")
                failed += 1
            except Exception as e:
                logger.critical(f"Unexpected error: {e}")
                failed += 1

        logger.info(f"Complete: {successful} successful, {failed} failed")

    print("\nProcessing trades with logging:")

    trades = [
        {"ticker": "AAPL", "price": "150.50", "shares": "100"},
        {"ticker": "GOOGL", "price": "invalid", "shares": "50"},
        {"ticker": "MSFT", "shares": "75"},  # Missing price
    ]

    process_trades(trades)


def main() -> None:
    """Run all error handling examples."""
    intro()
    basic_try_except()
    multiple_exceptions()
    finally_clause()
    custom_exceptions()
    else_clause()
    raising_exceptions()
    context_managers()
    assertion_checking()
    logging_errors()
    print("\n🎉 Error Handling tutorial complete!")
    print("Build robust, production-ready trading systems with proper error handling.")


if __name__ == "__main__":
    main()
