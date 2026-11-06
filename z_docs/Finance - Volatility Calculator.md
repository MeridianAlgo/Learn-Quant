# Volatility Calculator

<p class="lq-meta">Intermediate <span>/</span> Options, Derivatives & Finance <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Finance%20-%20Volatility%20Calculator">source</a></p>


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

---

## Running it

```bash
cd "Finance - Volatility Calculator"
python "volatility_calculator.py"
```

## More in Options, Derivatives & Finance

- [Advanced Options Pricing](Advanced Options Pricing.md): This module covers advanced mathematical techniques for pricing financial derivatives.
- [Black-Scholes Option Pricing](Black-Scholes Option Pricing.md): This module lets you price basic stock options (calls and puts) using the Black-Scholes formula, a foundation of modern financial analysis.
- [Bond Price and Yield](Bond Price and Yield.md): This utility lets you calculate the fair price of a bond or estimate its yield to maturity (YTM), two of the most basic (and important!) ideas in investing.
- [CAPM](CAPM.md): CAPM is the idea that won a Nobel Prize and still anchors how the industry thinks about return and risk.
- [Discounted Cash Flow (DCF)](Discounted Cash Flow (DCF).md): This tool calculates the present value of a series of future cash flows—the basic principle behind valuing businesses, real estate, projects, and stocks!
- [Dividend Tracker](Dividend Tracker.md): This tool lets you track upcoming dividends for a list of stocks, calculate expected income, and view a dividend calendar.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
