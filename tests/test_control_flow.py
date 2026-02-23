"""
Test suite for Python Basics - Control Flow utility.

Run with:
    python -m pytest tests/test_control_flow.py -v

Or directly:
    python tests/test_control_flow.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_conditional_logic():
    """Test if/elif/else conditional statements."""

    # Test risk level assessment
    def assess_risk(volatility: float) -> str:
        """Assess risk level based on volatility."""
        if volatility < 0.15:
            return "Low"
        elif volatility < 0.30:
            return "Medium"
        else:
            return "High"

    assert assess_risk(0.10) == "Low", "Low volatility should return Low risk"
    assert assess_risk(0.25) == "Medium", "Medium volatility should return Medium risk"
    assert assess_risk(0.35) == "High", "High volatility should return High risk"
    assert assess_risk(0.15) == "Medium", "Boundary case: 0.15 should be Medium"

    print("✓ Conditional logic tests passed")


def test_for_loop_iteration():
    """Test for loop iterations over collections."""
    # Test portfolio iteration
    portfolio = {"AAPL": 50, "GOOGL": 20, "MSFT": 30}

    total_shares = 0
    for _ticker, shares in portfolio.items():
        total_shares += shares

    assert total_shares == 100, "Total shares should equal 100"

    # Test cumulative returns calculation
    monthly_returns = [0.02, -0.01, 0.03, 0.015, -0.005]
    cumulative = 1.0

    for ret in monthly_returns:
        cumulative *= 1 + ret

    expected = 1.02 * 0.99 * 1.03 * 1.015 * 0.995
    assert abs(cumulative - expected) < 0.0001, "Cumulative return calculation incorrect"

    print("✓ For loop iteration tests passed")


def test_while_loop():
    """Test while loop functionality."""
    # Test investment doubling calculation
    initial_investment = 1000.0
    annual_return = 0.10
    target = initial_investment * 2

    years = 0
    current_value = initial_investment

    while current_value < target:
        years += 1
        current_value *= 1 + annual_return

    # Rule of 72: approx years = 72 / (return_pct)
    # At 10%, should take roughly 7.2 years
    assert 7 <= years <= 8, f"Investment should double in 7-8 years, got {years}"
    assert current_value >= target, "Final value should meet or exceed target"

    print("✓ While loop tests passed")


def test_list_comprehensions():
    """Test list comprehension functionality."""
    # Test percentage change calculations
    prices = [100, 102, 98, 101, 105, 103]

    pct_changes = [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]

    assert len(pct_changes) == 5, "Should have 5 price changes for 6 prices"
    assert abs(pct_changes[0] - 0.02) < 0.0001, "First change should be +2%"
    assert abs(pct_changes[1] - (-0.0392)) < 0.01, "Second change should be approximately -3.92%"

    # Test filtering profitable trades
    trades = [
        {"symbol": "AAPL", "pnl": 150.50},
        {"symbol": "GOOGL", "pnl": -75.25},
        {"symbol": "MSFT", "pnl": 200.00},
        {"symbol": "TSLA", "pnl": -50.00},
    ]

    profitable = [trade for trade in trades if trade["pnl"] > 0]

    assert len(profitable) == 2, "Should have 2 profitable trades"
    assert all(trade["pnl"] > 0 for trade in profitable), "All filtered trades should be profitable"

    # Test dictionary comprehension
    holdings = {"AAPL": 50, "GOOGL": 20}
    prices_dict = {"AAPL": 175.50, "GOOGL": 140.25}

    portfolio_values = {ticker: holdings[ticker] * prices_dict[ticker] for ticker in holdings}

    assert abs(portfolio_values["AAPL"] - 8775.0) < 0.01, "AAPL value incorrect"
    assert abs(portfolio_values["GOOGL"] - 2805.0) < 0.01, "GOOGL value incorrect"

    print("✓ List comprehension tests passed")


def test_break_statement():
    """Test break statement functionality."""
    # Test early exit when target reached
    daily_pnl = [50, 75, 120, 200, 150, 90, 60]
    target_profit = 300

    cumulative_pnl = 0
    days_to_target = 0

    for day, pnl in enumerate(daily_pnl, 1):
        cumulative_pnl += pnl
        days_to_target = day

        if cumulative_pnl >= target_profit:
            break

    assert days_to_target == 4, "Should reach target in 4 days"
    assert cumulative_pnl >= target_profit, "Should meet or exceed target"

    print("✓ Break statement tests passed")


def test_continue_statement():
    """Test continue statement functionality."""
    # Test skipping losing trades
    all_trades = [
        {"day": 1, "pnl": 100},
        {"day": 2, "pnl": -50},
        {"day": 3, "pnl": 150},
        {"day": 4, "pnl": -25},
    ]

    profitable_days = []

    for trade in all_trades:
        if trade["pnl"] < 0:
            continue
        profitable_days.append(trade["day"])

    assert profitable_days == [1, 3], "Should only include days 1 and 3"
    assert len(profitable_days) == 2, "Should have 2 profitable days"

    print("✓ Continue statement tests passed")


def test_nested_loops():
    """Test nested loop functionality."""
    # Test creating a simple correlation matrix
    tickers = ["AAPL", "GOOGL", "MSFT"]

    # Create identity matrix (each asset perfectly correlated with itself)
    correlation_matrix = []

    for i in range(len(tickers)):
        row = []
        for j in range(len(tickers)):
            if i == j:
                row.append(1.0)
            else:
                row.append(0.0)
        correlation_matrix.append(row)

    # Check diagonal is all 1.0
    for i in range(len(tickers)):
        assert correlation_matrix[i][i] == 1.0, f"Diagonal element [{i}][{i}] should be 1.0"

    # Check off-diagonal are 0.0
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            if i != j:
                assert correlation_matrix[i][j] == 0.0, f"Element [{i}][{j}] should be 0.0"

    print("✓ Nested loop tests passed")


def test_range_function():
    """Test range() function usage."""
    # Test basic range
    numbers = list(range(5))
    assert numbers == [0, 1, 2, 3, 4], "range(5) should produce 0-4"

    # Test range with start and stop
    numbers = list(range(1, 6))
    assert numbers == [1, 2, 3, 4, 5], "range(1, 6) should produce 1-5"

    # Test range with step
    numbers = list(range(0, 10, 2))
    assert numbers == [0, 2, 4, 6, 8], "range(0, 10, 2) should produce even numbers"

    print("✓ Range function tests passed")


def test_enumerate_function():
    """Test enumerate() function usage."""
    tickers = ["AAPL", "GOOGL", "MSFT"]

    indexed_tickers = []
    for index, ticker in enumerate(tickers):
        indexed_tickers.append((index, ticker))

    assert indexed_tickers[0] == (0, "AAPL"), "First element should be (0, 'AAPL')"
    assert indexed_tickers[2] == (2, "MSFT"), "Third element should be (2, 'MSFT')"

    # Test enumerate with custom start
    indexed_tickers = []
    for index, ticker in enumerate(tickers, start=1):
        indexed_tickers.append((index, ticker))

    assert indexed_tickers[0] == (
        1,
        "AAPL",
    ), "First element should be (1, 'AAPL') with start=1"

    print("✓ Enumerate function tests passed")


def run_all_tests():
    """Run all test functions."""
    print("\n" + "=" * 60)
    print("Testing Python Basics - Control Flow")
    print("=" * 60 + "\n")

    test_conditional_logic()
    test_for_loop_iteration()
    test_while_loop()
    test_list_comprehensions()
    test_break_statement()
    test_continue_statement()
    test_nested_loops()
    test_range_function()
    test_enumerate_function()

    print("\n" + "=" * 60)
    print("All Control Flow tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
