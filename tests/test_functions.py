"""
Test suite for Python Basics - Functions utility.

Run with:
    python -m pytest tests/test_functions.py -v

Or directly:
    python tests/test_functions.py
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_basic_function_definition():
    """Test basic function definitions and calls."""

    def calculate_profit(buy_price: float, sell_price: float, shares: int) -> float:
        """Calculate profit from a trade."""
        return (sell_price - buy_price) * shares

    profit = calculate_profit(100, 105, 50)
    assert profit == 250, "Profit calculation incorrect"

    profit_negative = calculate_profit(100, 95, 50)
    assert profit_negative == -250, "Loss calculation incorrect"

    print("✓ Basic function definition tests passed")


def test_default_parameters():
    """Test functions with default parameters."""

    def calculate_position_size(
        account_balance: float, risk_percent: float = 0.02
    ) -> float:
        """Calculate position size with default 2% risk."""
        return account_balance * risk_percent

    size1 = calculate_position_size(10000)
    assert size1 == 200, "Default parameter not working"

    size2 = calculate_position_size(10000, 0.05)
    assert size2 == 500, "Custom parameter not working"

    print("✓ Default parameter tests passed")


def test_multiple_return_values():
    """Test functions returning multiple values."""

    def calculate_metrics(prices: List[float]) -> Tuple[float, float, float]:
        """Return min, max, and average."""
        return min(prices), max(prices), sum(prices) / len(prices)

    prices = [100, 105, 98, 110, 102]
    min_val, max_val, avg_val = calculate_metrics(prices)

    assert min_val == 98, "Min value incorrect"
    assert max_val == 110, "Max value incorrect"
    assert abs(avg_val - 103) < 0.01, "Average value incorrect"

    print("✓ Multiple return value tests passed")


def test_args_variable_arguments():
    """Test *args for variable positional arguments."""

    def calculate_total(*values: float) -> float:
        """Sum variable number of values."""
        return sum(values)

    total1 = calculate_total(100, 200, 300)
    assert total1 == 600, "*args with 3 arguments failed"

    total2 = calculate_total(100, 200, 300, 400, 500)
    assert total2 == 1500, "*args with 5 arguments failed"

    total3 = calculate_total(100)
    assert total3 == 100, "*args with 1 argument failed"

    print("✓ *args variable arguments tests passed")


def test_kwargs_keyword_arguments():
    """Test **kwargs for variable keyword arguments."""

    def create_order(symbol: str, **kwargs) -> Dict:
        """Create order with optional parameters."""
        order = {"symbol": symbol}
        order.update(kwargs)
        return order

    order1 = create_order("AAPL", quantity=100, price=150.00)
    assert order1["symbol"] == "AAPL", "Symbol not set"
    assert order1["quantity"] == 100, "Quantity not set"
    assert order1["price"] == 150.00, "Price not set"

    order2 = create_order("GOOGL", quantity=50)
    assert "quantity" in order2, "Quantity missing"
    assert "price" not in order2, "Price should not be present"

    print("✓ **kwargs keyword arguments tests passed")


def test_lambda_functions():
    """Test lambda (anonymous) functions."""
    # Simple lambda
    square = lambda x: x**2
    assert square(5) == 25, "Lambda square function failed"

    # Lambda with filter
    prices = [45, 120, 85, 200, 30, 150]
    high_prices = list(filter(lambda x: x > 100, prices))
    assert high_prices == [120, 200, 150], "Lambda filter failed"

    # Lambda with map
    returns = [0.02, -0.01, 0.03]
    scaled_returns = [x * 100 for x in returns]
    assert scaled_returns == [2.0, -1.0, 3.0], "Lambda map failed"

    # Lambda with sorted
    portfolio = [
        {"ticker": "AAPL", "value": 5000},
        {"ticker": "GOOGL", "value": 3000},
        {"ticker": "MSFT", "value": 7000},
    ]
    sorted_portfolio = sorted(portfolio, key=lambda x: x["value"], reverse=True)
    assert sorted_portfolio[0]["ticker"] == "MSFT", "Lambda sort failed"

    print("✓ Lambda function tests passed")


def test_function_as_parameter():
    """Test passing functions as parameters (higher-order functions)."""

    def apply_to_prices(prices: List[float], transform_func) -> List[float]:
        """Apply transformation function to all prices."""
        return [transform_func(p) for p in prices]

    prices = [100, 200, 300]

    # Double all prices
    doubled = apply_to_prices(prices, lambda x: x * 2)
    assert doubled == [200, 400, 600], "Function parameter failed"

    # Add 10% to all prices
    increased = apply_to_prices(prices, lambda x: x * 1.10)
    assert abs(increased[0] - 110) < 0.01, "Function parameter transformation failed"

    print("✓ Function as parameter tests passed")


def test_nested_functions():
    """Test nested function definitions."""

    def outer_function(multiplier: float):
        """Outer function that returns inner function."""

        def inner_function(value: float) -> float:
            """Inner function that uses outer variable."""
            return value * multiplier

        return inner_function

    double = outer_function(2)
    triple = outer_function(3)

    assert double(10) == 20, "Nested function with multiplier 2 failed"
    assert triple(10) == 30, "Nested function with multiplier 3 failed"

    print("✓ Nested function tests passed")


def test_optional_return_values():
    """Test functions with optional return values."""

    def find_ticker(symbol: str, watchlist: List[str]) -> Optional[str]:
        """Find ticker in watchlist, return None if not found."""
        if symbol in watchlist:
            return symbol
        return None

    watchlist = ["AAPL", "GOOGL", "MSFT"]

    result1 = find_ticker("AAPL", watchlist)
    assert result1 == "AAPL", "Should find AAPL"

    result2 = find_ticker("TSLA", watchlist)
    assert result2 is None, "Should return None for TSLA"

    print("✓ Optional return value tests passed")


def test_recursive_functions():
    """Test recursive function calls."""

    def calculate_compound(principal: float, rate: float, years: int) -> float:
        """Calculate compound interest recursively."""
        # Base case
        if years == 0:
            return principal
        # Recursive case
        return calculate_compound(principal * (1 + rate), rate, years - 1)

    result = calculate_compound(1000, 0.10, 3)
    expected = 1000 * (1.10**3)
    assert abs(result - expected) < 0.01, "Recursive compound interest failed"

    print("✓ Recursive function tests passed")


def test_function_scope():
    """Test variable scope in functions."""
    global_fee = 0.001

    def calculate_net_profit(gross_profit: float) -> float:
        """Use global variable in function."""
        fee = gross_profit * global_fee
        return gross_profit - fee

    net = calculate_net_profit(1000)
    assert abs(net - 999) < 0.01, "Global variable scope failed"

    # Test local variable doesn't affect global
    def use_local_fee(gross_profit: float) -> float:
        """Use local variable."""
        local_fee = 0.002
        fee = gross_profit * local_fee
        return gross_profit - fee

    net_local = use_local_fee(1000)
    assert abs(net_local - 998) < 0.01, "Local variable calculation failed"

    print("✓ Function scope tests passed")


def test_type_hints():
    """Test that type hints work correctly."""

    def calculate_return(start_price: float, end_price: float) -> float:
        """Calculate return with type hints."""
        return (end_price - start_price) / start_price

    # Type hints don't enforce types at runtime in Python,
    # but we can still test the function works correctly
    result = calculate_return(100.0, 110.0)
    assert abs(result - 0.10) < 0.0001, "Type-hinted function failed"

    # Even with wrong types, Python will try to make it work
    result_int = calculate_return(100, 110)
    assert abs(result_int - 0.10) < 0.0001, "Type-hinted function with ints failed"

    print("✓ Type hint tests passed")


def run_all_tests():
    """Run all test functions."""
    print("\n" + "=" * 60)
    print("Testing Python Basics - Functions")
    print("=" * 60 + "\n")

    test_basic_function_definition()
    test_default_parameters()
    test_multiple_return_values()
    test_args_variable_arguments()
    test_kwargs_keyword_arguments()
    test_lambda_functions()
    test_function_as_parameter()
    test_nested_functions()
    test_optional_return_values()
    test_recursive_functions()
    test_function_scope()
    test_type_hints()

    print("\n" + "=" * 60)
    print("All Functions tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
