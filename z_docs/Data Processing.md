<p class="lq-badges"><span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-cat">Utilities & Tools</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Data Processing"
    python "data_validation_utils.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Data%20Processing)

---
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


---

## Continue in Utilities & Tools

<div class="grid cards" markdown>

-   :material-tools: __[Core Utilities](Core Utilities.md)__

    This folder contains core mathematical and date/time utilities that form the foundation for quantitative finance calculations.

-   :material-tools: __[Currency Converter](Currency Converter.md)__

    **This utility does NOT use any external APIs.** All exchange rates are entered manually for learning and experimentation.

-   :material-tools: __[Economic Calendar](Economic Calendar.md)__

    **This utility does NOT use any external APIs.** All data is managed locally for learning and experimentation.

-   :material-tools: __[Historical Data](Historical Data.md)__

    A Node.js script that fetches historical bars (OHLCV data) for stocks or crypto from the Alpaca Market Data API. It prompts interactively for the symbol type, symbol, timeframe, and date range, then prints the results as JSON.

-   :material-tools: __[Logging](Logging.md)__

    A pair of minimal, dependency-light logging utilities implemented in both Python and JavaScript. Each supports adding, reading, editing, and deleting log entries through an interactive command-line menu. All entries are persisted to a plain-text `log.txt` file in the working directory.

-   :material-tools: __[Market Data](Market Data.md)__

    This folder contains utilities for processing, analyzing, and fetching market data for financial applications.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
