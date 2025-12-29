# String Manipulation Utilities

This module provides comprehensive string manipulation utilities for financial applications, including case conversion, string truncation, number extraction, currency formatting, and URL slug generation.

## Functions

### `camel_to_snake(camel_str: str) -> str`
Converts camelCase string to snake_case.

**Parameters:**
- `camel_str`: CamelCase string

**Returns:**
- snake_case string

**Example:**
```python
>>> camel_to_snake("calculateProfit")
"calculate_profit"
>>> camel_to_snake("portfolioValue")
"portfolio_value"
```

### `snake_to_camel(snake_str: str) -> str`
Converts snake_case string to camelCase.

**Parameters:**
- `snake_str`: snake_case string

**Returns:**
- camelCase string

**Example:**
```python
>>> snake_to_camel("calculate_profit")
"calculateProfit"
>>> snake_to_camel("portfolio_value")
"portfolioValue"
```

### `truncate_string(text: str, max_length: int, suffix: str = '...') -> str`
Truncates string to specified length with suffix.

**Parameters:**
- `text`: String to truncate
- `max_length`: Maximum length
- `suffix`: Suffix to add if truncated

**Returns:**
- Truncated string

**Example:**
```python
>>> truncate_string("This is a very long string", 20)
"This is a very lo..."
>>> truncate_string("Short string", 20)
"Short string"
```

### `extract_numbers(text: str) -> List[float]`
Extracts all numbers from a string.

**Parameters:**
- `text`: String to extract numbers from

**Returns:**
- List of numbers found

**Example:**
```python
>>> extract_numbers("AAPL: $150.25, Volume: 1,234,567")
[150.25, 1234567.0]
>>> extract_numbers("Price increased by 2.5% to 102.50")
[2.5, 102.5]
```

### `clean_currency_string(currency_str: str) -> float`
Cleans currency string and converts to float.

**Parameters:**
- `currency_str`: Currency string (e.g., "$1,234.56")

**Returns:**
- Numeric value

**Raises:**
- `ValueError`: If invalid currency format

**Example:**
```python
>>> clean_currency_string("$1,234.56")
1234.56
>>> clean_currency_string("€2.500,50")
2500.50
```

### `format_currency(amount: float, currency: str = 'USD') -> str`
Formats amount as currency string.

**Parameters:**
- `amount`: Amount to format
- `currency`: Currency code

**Returns:**
- Formatted currency string

**Example:**
```python
>>> format_currency(1234.56)
"$1,234.56"
>>> format_currency(1234.56, "EUR")
"1,234.56 EUR"
```

### `generate_slug(text: str) -> str`
Generates URL-friendly slug from text.

**Parameters:**
- `text`: Text to convert to slug

**Returns:**
- URL-friendly slug

**Example:**
```python
>>> generate_slug("Apple Inc. Stock Analysis")
"apple-inc-stock-analysis"
>>> generate_slug("Trading Strategy for 2024")
"trading-strategy-for-2024"
```

## Usage

```python
from string_utils import (
    camel_to_snake, snake_to_camel, truncate_string, extract_numbers,
    clean_currency_string, format_currency, generate_slug
)

# Convert between naming conventions
api_field = "portfolioValue"
db_field = camel_to_snake(api_field)
print(f"API field {api_field} -> DB field {db_field}")

# Truncate long descriptions
description = "This is a very long stock description that needs to be truncated for display purposes"
short_desc = truncate_string(description, 50)
print(f"Shortened: {short_desc}")

# Extract numbers from financial news
news_text = "Apple stock rose to $150.25 with volume of 1,234,567 shares, up 2.5% from yesterday"
numbers = extract_numbers(news_text)
print(f"Extracted numbers: {numbers}")

# Clean and format currency
price_str = "$1,234.56"
price_num = clean_currency_string(price_str)
formatted = format_currency(price_num)
print(f"Cleaned: {price_num}, Formatted: {formatted}")

# Generate URL slugs for articles
title = "Top 10 Trading Strategies for Beginners"
slug = generate_slug(title)
print(f"Article slug: {slug}")
```

## Installation

No additional dependencies required. Uses only Python standard library.

## Testing

Run the module directly to see demonstrations:

```bash
python string_utils.py
```

## Common Use Cases

- **Data Transformation**: Convert between API and database naming conventions
- **Display Formatting**: Truncate long text for UI display
- **Data Extraction**: Extract numeric values from financial reports
- **Currency Processing**: Clean and format monetary values
- **URL Generation**: Create SEO-friendly URLs for articles and reports
- **Text Processing**: Clean and normalize user input
- **Report Generation**: Format financial data for presentation

## Notes

- Currency formatting currently supports USD format with automatic comma placement
- For other currencies, the amount is formatted with commas and currency code appended
- Number extraction handles integers, decimals, and negative numbers
- Slug generation removes special characters and converts to lowercase
- String truncation preserves word boundaries when possible
