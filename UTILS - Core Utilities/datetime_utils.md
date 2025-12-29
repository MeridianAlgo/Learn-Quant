# Date/Time Utilities

This module provides comprehensive date and time utilities for financial applications, including timestamp generation, flexible date parsing, trading day calculations, market hours checking, and duration formatting.

## Functions

### `get_current_timestamp() -> str`
Gets current timestamp in ISO format.

**Returns:**
- Current timestamp string

**Example:**
```python
>>> get_current_timestamp()
"2024-01-15T10:30:45.123456"
```

### `parse_flexible_date(date_string: str) -> datetime`
Parses date string with flexible formats.

**Parameters:**
- `date_string`: Date string to parse

**Returns:**
- Parsed datetime object

**Raises:**
- `ValueError`: If date cannot be parsed

**Example:**
```python
>>> parse_flexible_date("2024-01-15")
datetime.datetime(2024, 1, 15, 0, 0)
>>> parse_flexible_date("01/15/2024")
datetime.datetime(2024, 1, 15, 0, 0)
```

### `get_trading_days(start_date: datetime, end_date: datetime) -> List[datetime]`
Gets list of trading days (weekdays) between two dates.

**Parameters:**
- `start_date`: Start date
- `end_date`: End date

**Returns:**
- List of trading days (weekdays only)

**Example:**
```python
>>> from datetime import datetime
>>> start = datetime(2024, 1, 15)
>>> end = datetime(2024, 1, 19)
>>> trading_days = get_trading_days(start, end)
>>> len(trading_days)
5  # Monday to Friday
```

### `is_market_open(timestamp: datetime = None) -> bool`
Checks if market is currently open (9:30 AM - 4:00 PM EST, weekdays).

**Parameters:**
- `timestamp`: Timestamp to check (default: current time)

**Returns:**
- True if market is open, False otherwise

**Example:**
```python
>>> is_market_open()
False  # Assuming it's outside market hours
>>> is_market_open(datetime(2024, 1, 15, 10, 30))
True  # 10:30 AM on a weekday
```

### `get_next_trading_day(date: datetime = None) -> datetime`
Gets the next trading day (weekday).

**Parameters:**
- `date`: Reference date (default: current date)

**Returns:**
- Next trading day

**Example:**
```python
>>> from datetime import datetime
>>> friday = datetime(2024, 1, 12)  # Friday
>>> next_day = get_next_trading_day(friday)
>>> next_day.weekday()
0  # Monday
```

### `format_duration(seconds: float) -> str`
Formats duration in seconds to human-readable string.

**Parameters:**
- `seconds`: Duration in seconds

**Returns:**
- Human-readable duration string

**Example:**
```python
>>> format_duration(45)
"45.0 seconds"
>>> format_duration(125)
"2.1 minutes"
>>> format_duration(3661)
"1.0 hours"
>>> format_duration(90000)
"1.0 days"
```

## Usage

```python
from datetime_utils import (
    get_current_timestamp, parse_flexible_date, get_trading_days,
    is_market_open, get_next_trading_day, format_duration
)
from datetime import datetime, timedelta

# Get current timestamp for logging
timestamp = get_current_timestamp()

# Parse various date formats from user input
date1 = parse_flexible_date("2024-01-15")
date2 = parse_flexible_date("01/15/2024")

# Calculate trading days for analysis
end_date = datetime.now()
start_date = end_date - timedelta(days=30)
trading_days = get_trading_days(start_date, end_date)

# Check if market is open before placing trades
if is_market_open():
    print("Market is open - ready to trade!")
else:
    next_open = get_next_trading_day()
    print(f"Market closed. Next trading day: {next_open}")

# Format execution time for reports
execution_time = 125.5  # seconds
print(f"Strategy execution took: {format_duration(execution_time)}")
```

## Installation

No additional dependencies required. Uses only Python standard library.

## Testing

Run the module directly to see demonstrations:

```bash
python datetime_utils.py
```

## Common Use Cases

- **Market Timing**: Check if market is open before executing trades
- **Data Analysis**: Filter trading data to only include trading days
- **Logging**: Add timestamps to log entries and trade records
- **User Input**: Parse various date formats from user input
- **Performance Reporting**: Format execution times and durations
- **Scheduling**: Calculate next trading day for scheduled tasks
- **Historical Analysis**: Generate lists of trading days for backtesting

## Notes

- Market hours are based on US stock market (9:30 AM - 4:00 PM EST)
- Weekends are excluded from trading days
- Holidays are not accounted for (you may need to add holiday logic)
- All times are in local system time unless specified otherwise
