# Data Processing Utilities

This folder contains utilities for data processing, validation, and manipulation in financial applications.

## Available Utilities

### Data Validation (`data_validation_utils.py`)
- Email, phone number, and stock symbol validation
- Date format validation
- Numeric range validation
- String sanitization

### String Manipulation (`string_utils.py`)
- Case conversion (camelCase ↔ snake_case)
- String truncation
- Number extraction from text
- Currency formatting and cleaning
- URL slug generation

## Usage

```python
# Data validation
from data_validation_utils import validate_email, validate_stock_symbol
from string_utils import camel_to_snake, format_currency

# Validate inputs
if validate_email("user@example.com"):
    print("Valid email")

# Convert naming conventions
snake_case = camel_to_snake("calculateProfit")
print(snake_case)  # "calculate_profit"

# Format currency
formatted = format_currency(1234.56)
print(formatted)  # "$1,234.56"
```

## Installation

No additional dependencies required. Uses only Python standard library.

## Testing

Run each utility directly to see demonstrations:

```bash
python data_validation_utils.py
python string_utils.py
```

## Common Use Cases

- **Input Validation**: Validate user input in trading applications
- **Data Cleaning**: Clean and normalize financial data
- **String Processing**: Format text for reports and displays
- **Currency Handling**: Process monetary values correctly
- **API Integration**: Validate data from external APIs
