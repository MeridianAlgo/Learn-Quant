# Volatility Calculator

Calculate various volatility metrics for financial instruments.

## Features

- Historical Volatility (close-to-close)
- Parkinson Volatility (high-low estimator)
- Garman-Klass Volatility (OHLC estimator)
- EWMA Volatility (RiskMetrics)
- Realized Volatility (high-frequency)
- Volatility Cone Analysis

## Usage

```python
from volatility_calculator import historical_volatility, volatility_cone

prices = [100, 102, 101, 103, 105, 104, 106]
vol = historical_volatility(prices, window=5)
print(f"Volatility: {vol:.2%}")

cone = volatility_cone(prices)
```

## Methods

### Historical Volatility
Standard deviation of log returns, annualized to 252 trading days.

### Parkinson Volatility
Uses high-low range, more efficient than close-to-close.

### Garman-Klass Volatility
Most efficient OHLC estimator, accounts for opening jumps.

### EWMA Volatility
Exponentially weighted moving average, gives more weight to recent data.

### Volatility Cone
Shows volatility distribution across different time horizons.
