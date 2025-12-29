"""
Data Validation Utilities for Financial Applications

This module provides comprehensive data validation functions for financial applications,
including email validation, phone number validation, stock symbol validation, date validation,
numeric range validation, and string sanitization.

Author: Generated for Learn-Quant Project
Version: 1.0.0
"""

import re
from typing import Union


def validate_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid email format, False otherwise
        
    Example:
        >>> validate_email("user@example.com")
        True
        >>> validate_email("invalid.email")
        False
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_phone_number(phone: str) -> bool:
    """
    Validate phone number format (supports various formats).
    
    Args:
        phone: Phone number to validate
        
    Returns:
        True if valid phone format, False otherwise
        
    Example:
        >>> validate_phone_number("(555) 123-4567")
        True
        >>> validate_phone_number("555-1234")
        False
    """
    # Remove common formatting characters
    cleaned = re.sub(r'[\s\-\(\)]', '', phone)
    # Check if it's all digits and reasonable length (10-15 digits)
    return cleaned.isdigit() and 10 <= len(cleaned) <= 15


def validate_stock_symbol(symbol: str) -> bool:
    """
    Validate stock ticker symbol format.
    
    Args:
        symbol: Stock symbol to validate
        
    Returns:
        True if valid symbol format, False otherwise
        
    Example:
        >>> validate_stock_symbol("AAPL")
        True
        >>> validate_stock_symbol("GOOGL")
        True
        >>> validate_stock_symbol("INVALID123")
        False
    """
    # Most stock symbols are 1-5 letters, may include dots for some exchanges
    pattern = r'^[A-Z]{1,5}(\.[A-Z]{1,3})?$'
    return re.match(pattern, symbol.upper()) is not None


def validate_date(date_string: str, date_format: str = '%Y-%m-%d') -> bool:
    """
    Validate date string format.
    
    Args:
        date_string: Date string to validate
        date_format: Expected date format (default: YYYY-MM-DD)
        
    Returns:
        True if valid date format, False otherwise
        
    Example:
        >>> validate_date("2024-01-15")
        True
        >>> validate_date("01/15/2024", "%m/%d/%Y")
        True
        >>> validate_date("invalid date")
        False
    """
    from datetime import datetime
    try:
        datetime.strptime(date_string, date_format)
        return True
    except ValueError:
        return False


def validate_numeric_range(value: Union[int, float], min_val: float, max_val: float) -> bool:
    """
    Validate that a numeric value is within specified range.
    
    Args:
        value: Numeric value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        True if value is within range, False otherwise
        
    Example:
        >>> validate_numeric_range(50, 0, 100)
        True
        >>> validate_numeric_range(150, 0, 100)
        False
    """
    return min_val <= value <= max_val


def sanitize_string(input_string: str, max_length: int = 1000) -> str:
    """
    Sanitize string input by removing potentially harmful characters.
    
    Args:
        input_string: String to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
        
    Example:
        >>> sanitize_string("<script>alert('xss')</script>Hello World")
        "Hello World"
        >>> sanitize_string("   Too much    whitespace   ")
        "Too much whitespace"
    """
    if not input_string:
        return ""
    
    # Remove HTML tags
    clean = re.sub(r'<[^>]+>', '', input_string)
    # Remove excessive whitespace
    clean = re.sub(r'\s+', ' ', clean).strip()
    # Truncate if too long
    return clean[:max_length]


def demo_data_validation():
    """Demonstrate data validation functions."""
    print("=" * 60)
    print("DATA VALIDATION UTILITIES DEMONSTRATION")
    print("=" * 60)
    
    # Email validation
    print("\nEmail Validation:")
    test_emails = ["user@example.com", "invalid.email", "test@domain.co.uk", "bad@.com"]
    for email in test_emails:
        result = validate_email(email)
        print(f"  {email}: {'✓' if result else '✗'}")
    
    # Phone validation
    print("\nPhone Number Validation:")
    test_phones = ["(555) 123-4567", "555-123-4567", "5551234567", "123-456"]
    for phone in test_phones:
        result = validate_phone_number(phone)
        print(f"  {phone}: {'✓' if result else '✗'}")
    
    # Stock symbol validation
    print("\nStock Symbol Validation:")
    test_symbols = ["AAPL", "GOOGL", "BRK.B", "INVALID123", "TSLA"]
    for symbol in test_symbols:
        result = validate_stock_symbol(symbol)
        print(f"  {symbol}: {'✓' if result else '✗'}")
    
    # Date validation
    print("\nDate Validation:")
    test_dates = ["2024-01-15", "01/15/2024", "invalid date", "2024-13-45"]
    for date in test_dates:
        result = validate_date(date)
        print(f"  {date}: {'✓' if result else '✗'}")
    
    # Numeric range validation
    print("\nNumeric Range Validation (0-100):")
    test_values = [50, -10, 100, 150, 0]
    for value in test_values:
        result = validate_numeric_range(value, 0, 100)
        print(f"  {value}: {'✓' if result else '✗'}")
    
    # String sanitization
    print("\nString Sanitization:")
    test_strings = [
        "<script>alert('xss')</script>Hello World",
        "   Too much    whitespace   ",
        "Normal text with no issues",
        ""
    ]
    for text in test_strings:
        sanitized = sanitize_string(text)
        print(f"  Original: '{text}'")
        print(f"  Sanitized: '{sanitized}'")
        print()


def main():
    """Main function to run demonstrations."""
    demo_data_validation()
    print("\n🎉 Data validation utilities demonstration complete!")


if __name__ == "__main__":
    main()
