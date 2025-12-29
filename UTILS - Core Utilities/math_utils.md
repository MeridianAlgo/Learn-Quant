# Mathematical Utilities

This module provides comprehensive mathematical utilities for financial applications, including percentage calculations, compound interest, CAGR calculation, data normalization, moving averages, and linear regression.

## Functions

### `round_to_nearest(number: float, nearest: float) -> float`
Rounds number to nearest specified value.

**Parameters:**
- `number`: Number to round
- `nearest`: Value to round to (e.g., 0.05 for nickels)

**Returns:**
- Rounded number

**Example:**
```python
>>> round_to_nearest(1.23, 0.05)
1.25
>>> round_to_nearest(1.22, 0.05)
1.2
```

### `calculate_percentage_change(old_value: float, new_value: float) -> float`
Calculates percentage change between two values.

**Parameters:**
- `old_value`: Original value
- `new_value`: New value

**Returns:**
- Percentage change

**Raises:**
- `ValueError`: If original value is zero

**Example:**
```python
>>> calculate_percentage_change(100, 110)
10.0
>>> calculate_percentage_change(100, 90)
-10.0
```

### `compound_interest(principal: float, rate: float, periods: int, compound_frequency: int = 1) -> float`
Calculates compound interest.

**Parameters:**
- `principal`: Initial principal
- `rate`: Annual interest rate (as decimal)
- `periods`: Number of years
- `compound_frequency`: Times compounded per year

**Returns:**
- Final amount after compound interest

**Example:**
```python
>>> compound_interest(1000, 0.05, 5)
1276.28
>>> compound_interest(1000, 0.05, 5, 12)  # Monthly compounding
1283.36
```

### `calculate_cagr(beginning_value: float, ending_value: float, years: int) -> float`
Calculates Compound Annual Growth Rate (CAGR).

**Parameters:**
- `beginning_value`: Starting value
- `ending_value`: Ending value
- `years`: Number of years

**Returns:**
- CAGR as percentage

**Raises:**
- `ValueError`: If beginning value or years are not positive

**Example:**
```python
>>> calculate_cagr(1000, 1500, 3)
14.47
>>> calculate_cagr(10000, 20000, 7)
10.41
```

### `normalize_data(data: List[float], method: str = 'minmax') -> List[float]`
Normalizes data using specified method.

**Parameters:**
- `data`: List of numbers to normalize
- `method`: Normalization method ('minmax' or 'zscore')

**Returns:**
- Normalized data

**Raises:**
- `ValueError`: If unknown normalization method

**Example:**
```python
>>> normalize_data([10, 20, 30, 40, 50])
[0.0, 0.25, 0.5, 0.75, 1.0]
>>> normalize_data([10, 20, 30, 40, 50], 'zscore')
[-1.26, -0.63, 0.0, 0.63, 1.26]
```

### `moving_average(data: List[float], window: int) -> List[float]`
Calculates moving average.

**Parameters:**
- `data`: List of numbers
- `window`: Window size for moving average

**Returns:**
- List of moving averages

**Raises:**
- `ValueError`: If invalid window size

**Example:**
```python
>>> moving_average([1, 2, 3, 4, 5], 3)
[2.0, 3.0, 4.0]
>>> moving_average([10, 20, 30, 40], 2)
[15.0, 25.0, 35.0]
```

### `linear_regression(x: List[float], y: List[float]) -> Tuple[float, float]`
Performs simple linear regression.

**Parameters:**
- `x`: Independent variable values
- `y`: Dependent variable values

**Returns:**
- Tuple of (slope, intercept)

**Raises:**
- `ValueError`: If x and y have different lengths or insufficient data

**Example:**
```python
>>> linear_regression([1, 2, 3, 4], [2, 4, 6, 8])
(2.0, 0.0)
>>> linear_regression([1, 2, 3], [3, 5, 7])
(2.0, 1.0)
```

## Usage

```python
from math_utils import (
    round_to_nearest, calculate_percentage_change, compound_interest,
    calculate_cagr, normalize_data, moving_average, linear_regression
)

# Financial calculations
price_change = calculate_percentage_change(100, 110)
print(f"Stock price changed by {price_change:.2f}%")

# Investment growth
final_value = compound_interest(10000, 0.07, 10, 12)  # 7% annual, monthly compounding
print(f"10-year investment value: ${final_value:,.2f}")

# Portfolio performance
cagr = calculate_cagr(50000, 75000, 5)
print(f"Portfolio CAGR: {cagr:.2f}%")

# Data analysis
prices = [100, 105, 98, 110, 102, 108, 95]
normalized = normalize_data(prices)
ma_3day = moving_average(prices, 3)

# Trend analysis
months = list(range(1, 7))
revenue = [1000, 1200, 1150, 1400, 1350, 1600]
slope, intercept = linear_regression(months, revenue)
print(f"Revenue trend: ${slope:.2f} per month")
```

## Installation

No additional dependencies required. Uses only Python standard library.

## Testing

Run the module directly to see demonstrations:

```bash
python math_utils.py
```

## Common Use Cases

- **Financial Analysis**: Calculate returns, growth rates, and compound interest
- **Portfolio Management**: Normalize performance data and calculate trends
- **Technical Analysis**: Calculate moving averages and trend lines
- **Risk Management**: Analyze percentage changes and volatility
- **Data Processing**: Normalize datasets for comparison
- **Investment Planning**: Calculate future values and growth projections
- **Performance Metrics**: Calculate CAGR and other financial ratios

## Notes

- All monetary values should be provided as floats (not integers) for precision
- Percentage change calculation handles both positive and negative changes
- Linear regression uses ordinary least squares method
- Moving average returns fewer values than input data (window_size - 1 fewer)
- Z-score normalization assumes data follows normal distribution
- Compound interest can handle different compounding frequencies
