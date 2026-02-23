"""Advanced Python - Context Managers for Financial Applications.

Run with:
    python context_managers_tutorial.py

This module teaches context managers using the `with` statement,
creating custom context managers (functions and classes),
and applying them to financial use cases like timers and locks.
"""

import contextlib
import time
from typing import Generator, Optional


def intro() -> None:
    """Print orientation details."""
    print("\n" + "#" * 60)
    print("ADVANCED PYTHON – CONTEXT MANAGERS")
    print("#" * 60)
    print("Learn better resource management, error handling, and cleaner code")
    print("using the `with` statement and custom context managers.\n")


# -----------------------------------------------------------------------------
# 1. CLASS-BASED CONTEXT MANAGER
# -----------------------------------------------------------------------------


class Timer:
    """
    Measures the execution time of a code block.

    Demonstrates the class-based protocol: __enter__ and __exit__.
    """

    def __init__(self, label: str = "Operation"):
        """Initialize with a label."""
        self.label = label
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration: Optional[float] = None

    def __enter__(self) -> "Timer":
        """
        Start the timer when entering the context.

        Returns:
            self: Allows access to the timer object within the block
        """
        self.start_time = time.perf_counter()
        print(f"[{self.label}] Started...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Stop the timer when exiting the context.

        Args:
            exc_type: Exception type if raised
            exc_val: Exception value
            exc_tb: Traceback object

        Returns:
            False: Propagate exceptions if any occur
        """
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time

        status = "Failed" if exc_type else "Completed"
        print(f"[{self.label}] {status} in {self.duration:.6f} seconds")

        # Return True to suppress exception, False to propagate
        return False


def demonstrate_timer_class():
    """Demonstrate the class-based Timer context manager."""
    print("=" * 60)
    print("CLASS-BASED CONTEXT MANAGER: TIMER")
    print("=" * 60)

    # Example 1: Successful operation
    with Timer("DataProcessing") as t:
        print("  Processing heavy dataset...")
        time.sleep(0.5)  # Simulate work

    # Example 2: Accessing the object
    with Timer("Calculation") as t:
        print("  Calculating risks...")
        time.sleep(0.3)
    print(f"  > Accessed duration outside: {t.duration:.4f}s")

    print("\n")


# -----------------------------------------------------------------------------
# 2. FUNCTION-BASED CONTEXT MANAGER (contextlib)
# -----------------------------------------------------------------------------


@contextlib.contextmanager
def market_session(mock_open_time: str, mock_close_time: str) -> Generator[str, None, None]:
    """
    Simulate a market session.

    Demonstrates the generator-based approach with @contextlib.contextmanager.
    Setup code runs before 'yield', teardown runs after 'yield'.
    """
    print(f"🔔 Market OPEN at {mock_open_time}")
    status = "OPEN"

    try:
        # Pass control to the block inside 'with'
        yield status
    except Exception as e:
        print(f"🚨 Emergency HALT: {e}")
        # Re-raise unless you want to swallow the error
        raise
    finally:
        # This runs whether success or failure
        print(f"🔕 Market CLOSED at {mock_close_time}")


def demonstrate_contextlib():
    """Demonstrate function-based context managers."""
    print("=" * 60)
    print("FUNCTION-BASED CONTEXT MANAGER: @contextmanager")
    print("=" * 60)

    # Example 1: Normal session
    with market_session("09:30", "16:00") as status:
        print(f"  Trading is now {status}.")
        print("  Executing orders...")
        time.sleep(0.2)

    print()

    # Example 2: Handling Errors within the context manager
    try:
        with market_session("09:30", "16:00") as status:
            print(f"  Trading is {status}.")
            print("  Crash occurring...")
            raise RuntimeError("Flash Crash!")
    except RuntimeError:
        print("  > Exception caught outside context.")

    print("\n")


# -----------------------------------------------------------------------------
# 3. REAL-WORLD FINANCIAL EXAMPLE: ATOMIC TRANSACTION
# -----------------------------------------------------------------------------


class PortfolioTransaction:
    """
    Manages a portfolio update atomically.
    If an error occurs during the update, changes are rolled back.
    """

    def __init__(self, portfolio: dict):
        self.portfolio = portfolio
        self.backup = None

    def __enter__(self):
        # Save state before changes
        self.backup = self.portfolio.copy()
        print("💾 Backup created. Transaction started.")
        return self.portfolio

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            # Error occurred, rollback
            print(f"❌ Error encountered ({exc_val}). Rolling back...")
            self.portfolio.clear()
            self.portfolio.update(self.backup)
            print("↺ Rollback complete. State restored.")
            return True  # Suppress error for demo purposes

        print("✅ Transaction committed successfully.")
        return False


def demonstrate_atomic_transaction():
    """Demonstrate rollback capabilities."""
    print("=" * 60)
    print("FINANCIAL USE CASE: ATOMIC TRANSACTIONS")
    print("=" * 60)

    my_portfolio = {"AAPL": 10, "Cash": 1000}
    print(f"Initial State: {my_portfolio}")

    # Case 1: Successful Transaction
    print("\n--- Attempting Valid Trade ---")
    with PortfolioTransaction(my_portfolio) as p:
        p["Cash"] -= 150
        p["AAPL"] += 1
        print(f"  Inside CM: {p}")
    print(f"Final State: {my_portfolio}")

    # Case 2: Failed Transaction (Rollback)
    print("\n--- Attempting Invalid Trade (Crash) ---")
    with PortfolioTransaction(my_portfolio) as p:
        p["Cash"] -= 5000  # deducted
        print("  Cash deducted. Buying stock...")
        if p["Cash"] < 0:
            raise ValueError("Insufficient funds")
        p["AAPL"] += 50

    print(f"Final State:   {my_portfolio}")
    print("  (Notice Cash is back to original)")


def main() -> None:
    """Run all demonstrations."""
    intro()
    demonstrate_timer_class()
    demonstrate_contextlib()
    demonstrate_atomic_transaction()
    print("\n🎉 Context Managers tutorial complete!")
    print("Use them for resource cleanup, locks, timers, and transaction handling.")


if __name__ == "__main__":
    main()
