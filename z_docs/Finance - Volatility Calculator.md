<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Options, Derivatives & Finance</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Finance - Volatility Calculator"
    python "volatility_calculator.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Finance%20-%20Volatility%20Calculator)

---
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


---

## Continue in Options, Derivatives & Finance

<div class="grid cards" markdown>

-   :material-chart-bell-curve: __[Advanced Options Pricing](Advanced Options Pricing.md)__

    This module covers advanced mathematical techniques for pricing financial derivatives. The focus is on models beyond the standard assumptions. Rather than assuming constant volatility, we explore dynamic and local volatility models. These models are crucial for correctly valuing exotic options and managing the risks of complex derivatives portfolios.

-   :material-chart-bell-curve: __[Black-Scholes Option Pricing](Black-Scholes Option Pricing.md)__

    This module lets you price basic stock options (calls and puts) using the Black-Scholes formula, a foundation of modern financial analysis.

-   :material-chart-bell-curve: __[Bond Price and Yield](Bond Price and Yield.md)__

    This utility lets you calculate the fair price of a bond or estimate its yield to maturity (YTM), two of the most basic (and important!) ideas in investing.

-   :material-chart-bell-curve: __[CAPM](CAPM.md)__

    CAPM is the idea that won a Nobel Prize and still anchors how the industry

-   :material-chart-bell-curve: __[Discounted Cash Flow (DCF)](Discounted Cash Flow (DCF).md)__

    This tool calculates the present value of a series of future cash flows—the basic principle behind valuing businesses, real estate, projects, and stocks!

-   :material-chart-bell-curve: __[Dividend Tracker](Dividend Tracker.md)__

    **This utility does NOT use any external APIs.** All data is managed locally for learning and experimentation.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
