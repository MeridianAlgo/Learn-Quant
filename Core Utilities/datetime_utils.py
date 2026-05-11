"""
Date/Time Utilities for Financial Applications

This module provides comprehensive date and time utilities for financial applications,
including timestamp generation, flexible date parsing, trading day calculations,
market hours checking, and duration formatting.

Author: Generated for Learn-Quant Project
Version: 1.0.0
"""

from datetime import datetime, timedelta
from typing import List


def get_current_timestamp() -> str:
    """
    Get current timestamp in ISO format.

    Returns:
        Current timestamp string

    Example:
        >>> get_current_timestamp()
        "2024-01-15T10:30:45.123456"
    """
    return datetime.now().isoformat()


def parse_flexible_date(date_string: str) -> datetime:
    """
    Parse date string with flexible formats.

    Args:
        date_string: Date string to parse

    Returns:
        Parsed datetime object

    Raises:
        ValueError: If date cannot be parsed

    Example:
        >>> parse_flexible_date("2024-01-15")
        datetime.datetime(2024, 1, 15, 0, 0)
        >>> parse_flexible_date("01/15/2024")
        datetime.datetime(2024, 1, 15, 0, 0)
    """
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue

    raise ValueError(f"Unable to parse date: {date_string}")


def get_trading_days(start_date: datetime, end_date: datetime) -> List[datetime]:
    """
    Get list of trading days (weekdays) between two dates.

    Args:
        start_date: Start date
        end_date: End date

    Returns:
        List of trading days (weekdays only)

    Example:
        >>> from datetime import datetime
        >>> start = datetime(2024, 1, 15)
        >>> end = datetime(2024, 1, 19)
        >>> trading_days = get_trading_days(start, end)
        >>> len(trading_days)
        5  # Monday to Friday
    """
    trading_days = []
    current_date = start_date

    while current_date <= end_date:
        if current_date.weekday() < 5:  # Monday to Friday
            trading_days.append(current_date)
        current_date += timedelta(days=1)

    return trading_days


def is_market_open(timestamp: datetime = None) -> bool:
    """
    Check if market is currently open (9:30 AM - 4:00 PM EST, weekdays).

    Args:
        timestamp: Timestamp to check (default: current time)

    Returns:
        True if market is open, False otherwise

    Example:
        >>> is_market_open()
        False  # Assuming it's outside market hours
        >>> is_market_open(datetime(2024, 1, 15, 10, 30))
        True  # 10:30 AM on a weekday
    """
    if timestamp is None:
        timestamp = datetime.now()

    # Check if it's a weekday
    if timestamp.weekday() >= 5:  # Saturday or Sunday
        return False

    # Check if it's during market hours (9:30 AM - 4:00 PM EST)
    # Note: This doesn't account for holidays or time zones
    market_open = timestamp.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = timestamp.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= timestamp <= market_close


def get_next_trading_day(date: datetime = None) -> datetime:
    """
    Get the next trading day (weekday).

    Args:
        date: Reference date (default: current date)

    Returns:
        Next trading day

    Example:
        >>> from datetime import datetime
        >>> friday = datetime(2024, 1, 12)  # Friday
        >>> next_day = get_next_trading_day(friday)
        >>> next_day.weekday()
        0  # Monday
    """
    if date is None:
        date = datetime.now()

    next_day = date + timedelta(days=1)

    # Skip weekends
    while next_day.weekday() >= 5:  # Saturday or Sunday
        next_day += timedelta(days=1)

    return next_day


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Human-readable duration string

    Example:
        >>> format_duration(45)
        "45.0 seconds"
        >>> format_duration(125)
        "2.1 minutes"
        >>> format_duration(3661)
        "1.0 hours"
        >>> format_duration(90000)
        "1.0 days"
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f} hours"
    else:
        days = seconds / 86400
        return f"{days:.1f} days"


def demo_datetime_utils():
    """Demonstrate date/time utilities."""
    print("=" * 60)
    print("DATE/TIME UTILITIES DEMONSTRATION")
    print("=" * 60)

    # Current timestamp
    print("\nCurrent Timestamp:")
    timestamp = get_current_timestamp()
    print(f"  {timestamp}")

    # Flexible date parsing
    print("\nFlexible Date Parsing:")
    test_dates = [
        "2024-01-15",
        "01/15/2024",
        "2024-01-15 14:30:00",
        "2024-01-15T14:30:00",
    ]

    for date_str in test_dates:
        try:
            parsed = parse_flexible_date(date_str)
            print(f"  '{date_str}' -> {parsed}")
        except ValueError as e:
            print(f"  '{date_str}' -> Error: {e}")

    # Trading days
    print("\nTrading Days Calculation:")
    from datetime import datetime, timedelta

    end_date = datetime.now()
    start_date = end_date - timedelta(days=14)

    trading_days = get_trading_days(start_date, end_date)
    print(f"  Trading days in last 14 days: {len(trading_days)}")
    print(f"  First trading day: {trading_days[0].strftime('%Y-%m-%d')}")
    print(f"  Last trading day: {trading_days[-1].strftime('%Y-%m-%d')}")

    # Market hours
    print("\nMarket Hours Check:")
    market_status = is_market_open()
    print(f"  Is market currently open: {'Yes' if market_status else 'No'}")

    # Test specific times
    test_times = [
        datetime(2024, 1, 15, 9, 0),  # Before market open
        datetime(2024, 1, 15, 10, 30),  # During market hours
        datetime(2024, 1, 15, 16, 30),  # After market close
        datetime(2024, 1, 13, 10, 30),  # Saturday
    ]

    for test_time in test_times:
        is_open = is_market_open(test_time)
        day_name = test_time.strftime("%A")
        time_str = test_time.strftime("%H:%M")
        print(f"  {day_name} {time_str}: {'Open' if is_open else 'Closed'}")

    # Next trading day
    print("\nNext Trading Day:")
    next_trading = get_next_trading_day()
    print(f"  Next trading day: {next_trading.strftime('%Y-%m-%d (%A)')}")

    # Duration formatting
    print("\nDuration Formatting:")
    test_durations = [45, 125, 3661, 90000, 2592000]
    for duration in test_durations:
        formatted = format_duration(duration)
        print(f"  {duration:,} seconds -> {formatted}")


def main():
    """Main function to run demonstrations."""
    demo_datetime_utils()
    print("\n🎉 Date/Time utilities demonstration complete!")


if __name__ == "__main__":
    main()
