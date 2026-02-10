"""
Mathematical Utilities for Financial Applications

This module provides comprehensive mathematical utilities for financial applications,
including percentage calculations, compound interest, CAGR calculation, data normalization,
moving averages, and linear regression.

Author: Generated for Learn-Quant Project
Version: 1.0.0
"""

import math
from typing import List, Tuple


def round_to_nearest(number: float, nearest: float) -> float:
    """
    Round number to nearest specified value.

    Args:
        number: Number to round
        nearest: Value to round to (e.g., 0.05 for nickels)

    Returns:
        Rounded number

    Example:
        >>> round_to_nearest(1.23, 0.05)
        1.25
        >>> round_to_nearest(1.22, 0.05)
        1.2
    """
    return round(number / nearest) * nearest


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values.

    Args:
        old_value: Original value
        new_value: New value

    Returns:
        Percentage change

    Raises:
        ValueError: If original value is zero

    Example:
        >>> calculate_percentage_change(100, 110)
        10.0
        >>> calculate_percentage_change(100, 90)
        -10.0
    """
    if old_value == 0:
        raise ValueError("Original value cannot be zero")
    return ((new_value - old_value) / old_value) * 100


def compound_interest(principal: float, rate: float, periods: int, compound_frequency: int = 1) -> float:
    """
    Calculate compound interest.

    Args:
        principal: Initial principal
        rate: Annual interest rate (as decimal)
        periods: Number of years
        compound_frequency: Times compounded per year

    Returns:
        Final amount after compound interest

    Example:
        >>> compound_interest(1000, 0.05, 5)
        1276.28
        >>> compound_interest(1000, 0.05, 5, 12)  # Monthly compounding
        1283.36
    """
    return principal * (1 + rate / compound_frequency) ** (periods * compound_frequency)


def calculate_cagr(beginning_value: float, ending_value: float, years: int) -> float:
    """
    Calculate Compound Annual Growth Rate (CAGR).

    Args:
        beginning_value: Starting value
        ending_value: Ending value
        years: Number of years

    Returns:
        CAGR as percentage

    Raises:
        ValueError: If beginning value or years are not positive

    Example:
        >>> calculate_cagr(1000, 1500, 3)
        14.47
        >>> calculate_cagr(10000, 20000, 7)
        10.41
    """
    if beginning_value <= 0 or years <= 0:
        raise ValueError("Beginning value and years must be positive")

    return ((ending_value / beginning_value) ** (1 / years) - 1) * 100


def normalize_data(data: List[float], method: str = "minmax") -> List[float]:
    """
    Normalize data using specified method.

    Args:
        data: List of numbers to normalize
        method: Normalization method ('minmax' or 'zscore')

    Returns:
        Normalized data

    Raises:
        ValueError: If unknown normalization method

    Example:
        >>> normalize_data([10, 20, 30, 40, 50])
        [0.0, 0.25, 0.5, 0.75, 1.0]
        >>> normalize_data([10, 20, 30, 40, 50], 'zscore')
        [-1.26, -0.63, 0.0, 0.63, 1.26]
    """
    if not data:
        return []

    if method == "minmax":
        min_val, max_val = min(data), max(data)
        if max_val == min_val:
            return [0.0] * len(data)
        return [(x - min_val) / (max_val - min_val) for x in data]

    elif method == "zscore":
        mean_val = sum(data) / len(data)
        variance = sum((x - mean_val) ** 2 for x in data) / len(data)
        std_dev = math.sqrt(variance)

        if std_dev == 0:
            return [0.0] * len(data)
        return [(x - mean_val) / std_dev for x in data]

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def moving_average(data: List[float], window: int) -> List[float]:
    """
    Calculate moving average.

    Args:
        data: List of numbers
        window: Window size for moving average

    Returns:
        List of moving averages

    Raises:
        ValueError: If invalid window size

    Example:
        >>> moving_average([1, 2, 3, 4, 5], 3)
        [2.0, 3.0, 4.0]
        >>> moving_average([10, 20, 30, 40], 2)
        [15.0, 25.0, 35.0]
    """
    if window <= 0 or window > len(data):
        raise ValueError("Invalid window size")

    averages = []
    for i in range(len(data) - window + 1):
        window_avg = sum(data[i : i + window]) / window
        averages.append(window_avg)

    return averages


def linear_regression(x: List[float], y: List[float]) -> Tuple[float, float]:
    """
    Simple linear regression.

    Args:
        x: Independent variable values
        y: Dependent variable values

    Returns:
        Tuple of (slope, intercept)

    Raises:
        ValueError: If x and y have different lengths or insufficient data

    Example:
        >>> linear_regression([1, 2, 3, 4], [2, 4, 6, 8])
        (2.0, 0.0)
        >>> linear_regression([1, 2, 3], [3, 5, 7])
        (2.0, 1.0)
    """
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("x and y must have same length with at least 2 points")

    n = len(x)
    sum_x, sum_y = sum(x), sum(y)
    sum_xy, sum_x2 = sum(xi * yi for xi, yi in zip(x, y)), sum(xi**2 for xi in x)

    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    intercept = (sum_y - slope * sum_x) / n

    return slope, intercept


def demo_math_utils():
    """Demonstrate mathematical utilities."""
    print("=" * 60)
    print("MATHEMATICAL UTILITIES DEMONSTRATION")
    print("=" * 60)

    # Rounding
    print("\nRounding to Nearest:")
    numbers = [1.23, 1.27, 2.51, 2.49]
    for num in numbers:
        rounded = round_to_nearest(num, 0.05)
        print(f"  {num} -> {rounded} (nearest 0.05)")

    # Percentage change
    print("\nPercentage Change:")
    test_cases = [(100, 110), (100, 90), (50, 75), (200, 150)]
    for old, new in test_cases:
        change = calculate_percentage_change(old, new)
        print(f"  {old} -> {new}: {change:+.2f}%")

    # Compound interest
    print("\nCompound Interest:")
    principal = 10000
    rate = 0.07
    years = 10
    simple = compound_interest(principal, rate, years)
    monthly = compound_interest(principal, rate, years, 12)
    print(f"  Principal: ${principal:,}, Rate: {rate:.1%}, Years: {years}")
    print(f"  Annual compounding: ${simple:,.2f}")
    print(f"  Monthly compounding: ${monthly:,.2f}")
    print(f"  Difference: ${monthly - simple:,.2f}")

    # CAGR
    print("\nCompound Annual Growth Rate (CAGR):")
    investments = [(1000, 1500, 3), (10000, 20000, 7), (50000, 75000, 5)]
    for start, end, years in investments:
        cagr = calculate_cagr(start, end, years)
        print(f"  ${start:,} -> ${end:,} over {years} years: {cagr:.2f}% CAGR")

    # Data normalization
    print("\nData Normalization:")
    data = [10, 20, 30, 40, 50]
    minmax_norm = normalize_data(data, "minmax")
    zscore_norm = normalize_data(data, "zscore")
    print(f"  Original: {data}")
    print(f"  Min-Max: {[round(x, 3) for x in minmax_norm]}")
    print(f"  Z-Score: {[round(x, 3) for x in zscore_norm]}")

    # Moving average
    print("\nMoving Average:")
    prices = [100, 105, 98, 110, 102, 108, 95, 112]
    ma_3 = moving_average(prices, 3)
    ma_5 = moving_average(prices, 5)
    print(f"  Prices: {prices}")
    print(f"  3-day MA: {[round(x, 2) for x in ma_3]}")
    print(f"  5-day MA: {[round(x, 2) for x in ma_5]}")

    # Linear regression
    print("\nLinear Regression:")
    x_data = [1, 2, 3, 4, 5, 6]
    y_data = [1000, 1200, 1150, 1400, 1350, 1600]
    slope, intercept = linear_regression(x_data, y_data)
    print(f"  X: {x_data}")
    print(f"  Y: {y_data}")
    print(f"  Slope: {slope:.2f}, Intercept: {intercept:.2f}")
    print(f"  Equation: y = {slope:.2f}x + {intercept:.2f}")

    # Prediction example
    next_month = 7
    predicted = slope * next_month + intercept
    print(f"  Predicted value for month {next_month}: ${predicted:,.2f}")


def main():
    """Main function to run demonstrations."""
    demo_math_utils()
    print("\n🎉 Mathematical utilities demonstration complete!")


if __name__ == "__main__":
    main()
