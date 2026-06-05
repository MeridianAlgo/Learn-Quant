<p class="lq-badges"><span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-cat">Utilities & Tools</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Core Utilities"
    python "datetime_utils.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Core%20Utilities)

---
# Core Utilities

This folder contains core mathematical and date/time utilities that form the foundation for quantitative finance calculations.

## Available Utilities

### Date/Time (`datetime_utils.py`)
- Timestamp generation and parsing
- Trading day calculations
- Market hours checking
- Duration formatting

### Mathematical (`math_utils.py`)
- Percentage calculations
- Compound interest and CAGR
- Data normalization
- Moving averages
- Linear regression

## Usage

```python
# Date/time operations
from datetime_utils import get_trading_days, is_market_open, format_duration
from math_utils import calculate_cagr, moving_average, linear_regression

# Trading operations
trading_days = get_trading_days(start_date, end_date)
if is_market_open():
    print("Market is open!")

# Mathematical calculations
cagr = calculate_cagr(1000, 1500, 3)
ma = moving_average(prices, 20)
slope, intercept = linear_regression(x_data, y_data)
```

## Installation

No additional dependencies required. Uses only Python standard library.

## Testing

Run each utility directly to see demonstrations:

```bash
python datetime_utils.py
python math_utils.py
```

## Common Use Cases

- **Time Analysis**: Calculate trading days and market hours
- **Financial Calculations**: Perform core mathematical operations
- **Data Analysis**: Normalize and analyze time series data
- **Trend Analysis**: Calculate moving averages and regressions
- **Investment Planning**: Calculate compound growth and returns


---

## Continue in Utilities & Tools

<div class="grid cards" markdown>

-   :material-tools: __[Currency Converter](Currency Converter.md)__

    **This utility does NOT use any external APIs.** All exchange rates are entered manually for learning and experimentation.

-   :material-tools: __[Data Processing](Data Processing.md)__

    This folder contains utilities for data processing, validation, and manipulation in financial applications.

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
