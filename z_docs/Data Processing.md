# Data Processing Utilities

<p class="lq-meta">Beginner <span>/</span> Utilities & Tools <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Data%20Processing">source</a></p>


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

---

## Running it

```bash
cd "Data Processing"
python "data_validation_utils.py"
```

## More in Utilities & Tools

- [Core Utilities](Core Utilities.md): This folder contains core mathematical and date/time utilities that form the foundation for quantitative finance calculations.
- [Currency Converter](Currency Converter.md): This tool lets you convert between currencies using user-supplied exchange rates.
- [Economic Calendar](Economic Calendar.md): This tool lets you simulate and explore economic events (like FOMC meetings, CPI releases, etc.) and see how they might impact markets.
- [Historical Data](Historical Data.md): A Node.js script that fetches historical bars (OHLCV data) for stocks or crypto from the Alpaca Market Data API.
- [Logging](Logging.md): A pair of minimal, dependency-light logging utilities implemented in both Python and JavaScript.
- [Market Data](Market Data.md): This folder contains utilities for processing, analyzing, and fetching market data for financial applications.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
