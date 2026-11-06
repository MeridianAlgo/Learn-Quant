# Core Utilities

<p class="lq-meta">Beginner <span>/</span> Utilities & Tools <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Core%20Utilities">source</a></p>


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

## Running it

```bash
cd "Core Utilities"
python "datetime_utils.py"
```

## More in Utilities & Tools

- [Currency Converter](Currency Converter.md): This tool lets you convert between currencies using user-supplied exchange rates.
- [Data Processing](Data Processing.md): This folder contains utilities for data processing, validation, and manipulation in financial applications.
- [Economic Calendar](Economic Calendar.md): This tool lets you simulate and explore economic events (like FOMC meetings, CPI releases, etc.) and see how they might impact markets.
- [Historical Data](Historical Data.md): A Node.js script that fetches historical bars (OHLCV data) for stocks or crypto from the Alpaca Market Data API.
- [Logging](Logging.md): A pair of minimal, dependency-light logging utilities implemented in both Python and JavaScript.
- [Market Data](Market Data.md): This folder contains utilities for processing, analyzing, and fetching market data for financial applications.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
