# Data Validation Utilities

This module provides comprehensive data validation functions for financial applications, including email validation, phone number validation, stock symbol validation, date validation, numeric range validation, and string sanitization.

## Functions

### `validate_email(email: str) -> bool`
Validates email address format using regex pattern.

**Parameters:**
- `email`: Email address to validate

**Returns:**
- `True` if valid email format, `False` otherwise

**Example:**
```python
>>> validate_email("user@example.com")
True
>>> validate_email("invalid.email")
False
```

### `validate_phone_number(phone: str) -> bool`
Validates phone number format (supports various formats).

**Parameters:**
- `phone`: Phone number to validate

**Returns:**
- `True` if valid phone format, `False` otherwise

**Example:**
```python
>>> validate_phone_number("(555) 123-4567")
True
>>> validate_phone_number("555-1234")
False
```

### `validate_stock_symbol(symbol: str) -> bool`
Validates stock ticker symbol format.

**Parameters:**
- `symbol`: Stock symbol to validate

**Returns:**
- `True` if valid symbol format, `False` otherwise

**Example:**
```python
>>> validate_stock_symbol("AAPL")
True
>>> validate_stock_symbol("GOOGL")
True
>>> validate_stock_symbol("INVALID123")
False
```

### `validate_date(date_string: str, date_format: str = '%Y-%m-%d') -> bool`
Validates date string format.

**Parameters:**
- `date_string`: Date string to validate
- `date_format`: Expected date format (default: YYYY-MM-DD)

**Returns:**
- `True` if valid date format, `False` otherwise

**Example:**
```python
>>> validate_date("2024-01-15")
True
>>> validate_date("01/15/2024", "%m/%d/%Y")
True
>>> validate_date("invalid date")
False
```

### `validate_numeric_range(value: Union[int, float], min_val: float, max_val: float) -> bool`
Validates that a numeric value is within specified range.

**Parameters:**
- `value`: Numeric value to validate
- `min_val`: Minimum allowed value
- `max_val`: Maximum allowed value

**Returns:**
- `True` if value is within range, `False` otherwise

**Example:**
```python
>>> validate_numeric_range(50, 0, 100)
True
>>> validate_numeric_range(150, 0, 100)
False
```

### `sanitize_string(input_string: str, max_length: int = 1000) -> str`
Sanitizes string input by removing potentially harmful characters.

**Parameters:**
- `input_string`: String to sanitize
- `max_length`: Maximum allowed length

**Returns:**
- Sanitized string

**Example:**
```python
>>> sanitize_string("<script>alert('xss')</script>Hello World")
"Hello World"
>>> sanitize_string("   Too much    whitespace   ")
"Too much whitespace"
```

## Usage

```python
from data_validation_utils import (
    validate_email, validate_phone_number, validate_stock_symbol,
    validate_date, validate_numeric_range, sanitize_string
)

# Validate user input
email = "trader@example.com"
if validate_email(email):
    print("Valid email")

# Validate stock symbol
symbol = "AAPL"
if validate_stock_symbol(symbol):
    print(f"Valid symbol: {symbol}")

# Sanitize user input
user_input = "<script>alert('hack')</script>Trade data"
clean_input = sanitize_string(user_input)
```

## Installation

No additional dependencies required. Uses only Python standard library.

## Testing

Run the module directly to see demonstrations:

```bash
python data_validation_utils.py
```

## Common Use Cases

- **User Registration**: Validate email and phone numbers
- **Trading Applications**: Validate stock symbols and price ranges
- **Data Cleaning**: Sanitize user input and validate data formats
- **Form Validation**: Ensure user input meets required formats
- **API Input Validation**: Validate incoming API request parameters
