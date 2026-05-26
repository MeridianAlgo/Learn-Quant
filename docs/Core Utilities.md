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
