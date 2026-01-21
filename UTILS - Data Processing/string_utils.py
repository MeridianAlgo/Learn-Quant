"""
String Manipulation Utilities for Financial Applications

This module provides comprehensive string manipulation utilities for financial applications,
including case conversion, string truncation, number extraction, currency formatting,
and URL slug generation.

Author: Generated for Learn-Quant Project
Version: 1.0.0
"""

import re
from typing import List


def camel_to_snake(camel_str: str) -> str:
    """
    Convert camelCase string to snake_case.

    Args:
        camel_str: CamelCase string

    Returns:
        snake_case string

    Example:
        >>> camel_to_snake("calculateProfit")
        "calculate_profit"
        >>> camel_to_snake("portfolioValue")
        "portfolio_value"
    """
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", camel_str)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def snake_to_camel(snake_str: str) -> str:
    """
    Convert snake_case string to camelCase.

    Args:
        snake_str: snake_case string

    Returns:
        camelCase string

    Example:
        >>> snake_to_camel("calculate_profit")
        "calculateProfit"
        >>> snake_to_camel("portfolio_value")
        "portfolioValue"
    """
    components = snake_str.split("_")
    return components[0] + "".join(x.capitalize() for x in components[1:])


def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate string to specified length with suffix.

    Args:
        text: String to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated string

    Example:
        >>> truncate_string("This is a very long string", 20)
        "This is a very lo..."
        >>> truncate_string("Short string", 20)
        "Short string"
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix


def extract_numbers(text: str) -> List[float]:
    """
    Extract all numbers from a string.

    Args:
        text: String to extract numbers from

    Returns:
        List of numbers found

    Example:
        >>> extract_numbers("AAPL: $150.25, Volume: 1,234,567")
        [150.25, 1234567.0]
        >>> extract_numbers("Price increased by 2.5% to 102.50")
        [2.5, 102.5]
    """
    pattern = r"-?\d+\.?\d*"
    matches = re.findall(pattern, text)
    return [float(match) for match in matches]


def clean_currency_string(currency_str: str) -> float:
    """
    Clean currency string and convert to float.

    Args:
        currency_str: Currency string (e.g., "$1,234.56")

    Returns:
        Numeric value

    Raises:
        ValueError: If invalid currency format

    Example:
        >>> clean_currency_string("$1,234.56")
        1234.56
        >>> clean_currency_string("€2.500,50")
        2500.50
    """
    # Remove currency symbols, commas, and whitespace
    clean = re.sub(r"[$,\s€£¥]", "", currency_str)
    # Handle European decimal format (comma as decimal separator)
    if "." in clean and "," in clean:
        # If both dots and commas, assume comma is thousands separator
        clean = clean.replace(",", "")
    elif "," in clean and clean.count(",") == 1:
        # If only one comma, it might be decimal separator
        parts = clean.split(",")
        if len(parts) == 2 and len(parts[1]) <= 2:
            # Likely decimal separator
            clean = clean.replace(",", ".")
        else:
            # Likely thousands separator
            clean = clean.replace(",", "")

    try:
        return float(clean)
    except ValueError as e:
        raise ValueError(f"Invalid currency format: {currency_str}") from e


def format_currency(amount: float, currency: str = "USD") -> str:
    """
    Format amount as currency string.

    Args:
        amount: Amount to format
        currency: Currency code

    Returns:
        Formatted currency string

    Example:
        >>> format_currency(1234.56)
        "$1,234.56"
        >>> format_currency(1234.56, "EUR")
        "1,234.56 EUR"
    """
    if currency == "USD":
        return f"${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"


def generate_slug(text: str) -> str:
    """
    Generate URL-friendly slug from text.

    Args:
        text: Text to convert to slug

    Returns:
        URL-friendly slug

    Example:
        >>> generate_slug("Apple Inc. Stock Analysis")
        "apple-inc-stock-analysis"
        >>> generate_slug("Trading Strategy for 2024")
        "trading-strategy-for-2024"
    """
    # Convert to lowercase and replace spaces with hyphens
    slug = text.lower().replace(" ", "-")
    # Remove special characters except hyphens
    slug = re.sub(r"[^a-z0-9\-]", "", slug)
    # Remove multiple consecutive hyphens
    slug = re.sub(r"-+", "-", slug)
    # Remove leading/trailing hyphens
    return slug.strip("-")


def demo_string_utils():
    """Demonstrate string manipulation utilities."""
    print("=" * 60)
    print("STRING MANIPULATION UTILITIES DEMONSTRATION")
    print("=" * 60)

    # Case conversion
    print("\nCase Conversion:")
    camel_cases = ["calculateProfit", "portfolioValue", "stockPrice", "totalReturn"]
    for camel in camel_cases:
        snake = camel_to_snake(camel)
        back_to_camel = snake_to_camel(snake)
        print(f"  {camel} -> {snake} -> {back_to_camel}")

    # String truncation
    print("\nString Truncation:")
    long_texts = [
        "This is a very long string that needs to be truncated for display purposes",
        "Short string",
        "Medium length text here",
    ]
    for text in long_texts:
        truncated = truncate_string(text, 30)
        print(f"  '{text}' -> '{truncated}'")

    # Number extraction
    print("\nNumber Extraction:")
    financial_texts = [
        "AAPL: $150.25, Volume: 1,234,567, Change: +2.5%",
        "Revenue: $1.23B, EPS: $2.45, P/E: 28.5",
        "Price dropped from $200 to $180, a loss of $20",
    ]
    for text in financial_texts:
        numbers = extract_numbers(text)
        print(f"  '{text}'")
        print(f"    Numbers: {numbers}")

    # Currency cleaning and formatting
    print("\nCurrency Operations:")
    currency_strings = ["$1,234.56", "€2.500,50", "£1,000.00", "¥500"]
    for currency_str in currency_strings:
        try:
            clean_amount = clean_currency_string(currency_str)
            formatted = format_currency(clean_amount)
            print(f"  '{currency_str}' -> {clean_amount} -> '{formatted}'")
        except ValueError as e:
            print(f"  '{currency_str}' -> Error: {e}")

    # Slug generation
    print("\nSlug Generation:")
    titles = [
        "Apple Inc. Stock Analysis for 2024",
        "Top 10 Trading Strategies for Beginners",
        "Understanding Market Volatility & Risk Management",
        "How to Build a Diversified Portfolio",
    ]
    for title in titles:
        slug = generate_slug(title)
        print(f"  '{title}' -> '{slug}'")


def main():
    """Main function to run demonstrations."""
    demo_string_utils()
    print("\n🎉 String manipulation utilities demonstration complete!")


if __name__ == "__main__":
    main()
